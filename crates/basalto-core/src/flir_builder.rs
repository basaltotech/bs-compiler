// crates/basalto-core/src/flir_builder.rs
//
// Versão COMPLETA e CORRIGIDA — stencil 1D com memória compartilhada real,
// halo genérico para qualquer `radius`, bounds checks corretos.
//
// Este arquivo tem bastante comentário de Rust "básico" além dos comentários
// de lógica de GPU, já que é para você estudar enquanto lê.

use anyhow::{anyhow, Result};
use inkwell::{
    AddressSpace, IntPredicate,
    context::Context,
    targets::{Target, TargetMachine, TargetTriple, InitializationConfig, FileType},
    memory_buffer::MemoryBuffer,
};
use basalto_common::hardware::DeviceCapabilities;

// ============================================================================
// 1. REPRESENTAÇÃO FLIR
// ============================================================================
// `#[derive(...)]` gera automaticamente implementações de traits para a
// struct. Debug permite usar {:?} em println!/eprintln!. Clone permite
// copiar o valor. Serialize/Deserialize (do serde) permitem transformar
// isso em JSON e vice-versa — é assim que Python e Rust trocam essa
// estrutura via PyO3.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FlirOp {
    pub op: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub params: Option<serde_json::Value>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FlirModule {
    pub ops: Vec<FlirOp>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

// ============================================================================
// 2. BUILDER — gera o FlirModule a partir do grafo (placeholder por enquanto)
// ============================================================================
//
// `Result<FlirModule>` (do crate `anyhow`) é o tipo de retorno padrão para
// "isso pode dar certo (Ok) ou dar errado (Err)". Você vai ver muito `?` no
// código: `algo_que_pode_falhar()?` significa "se der erro, retorna esse
// erro imediatamente da função atual; se der certo, continua com o valor".
pub fn build_flir(_graph_str: &str, caps: &Option<DeviceCapabilities>) -> Result<FlirModule> {
    // TODO: parsear o grafo FX real vindo do PyTorch.
    // `_graph_str` está com underscore porque ainda não é usado — isso evita
    // um warning do compilador dizendo "variável não utilizada".

    // `caps.as_ref()` pega uma referência ao valor de dentro do Option, sem
    // "tomar posse" dele (não move o valor). `.map(...)` só executa a
    // closure se houver um valor (Some) — se for None, o resultado inteiro
    // vira None e o `.unwrap_or(128)` no final entrega 128 como padrão.
    let tile_size: i64 = caps.as_ref().map(|c| c.max_threads_per_block as i64).unwrap_or(128);

    // Radius fixo em 1 por enquanto — mas o código de geração de IR abaixo
    // já suporta qualquer valor de radius corretamente.
    let radius: i64 = 1;

    // f64 = 8 bytes por elemento. A tile tem tile_size elementos "úteis"
    // mais 2*radius elementos de halo (radius de cada lado).
    let shared_mem_bytes: u32 = ((tile_size + 2 * radius) as u32) * 8;

    let ops = vec![FlirOp {
        op: "stencil_1d".to_string(),
        inputs: vec!["x".to_string()],
        output: "y".to_string(),
        params: Some(serde_json::json!({
            "radius": radius,
            "coeffs": [0.2, 0.3, 0.5],
            "tile_size": tile_size,
            "shared_mem_bytes": shared_mem_bytes,
        })),
    }];

    Ok(FlirModule {
        ops,
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
    })
}

// ============================================================================
// 3. GERAÇÃO DE LLVM IR
// ============================================================================
//
// Geometria de memória usada aqui (importante entender antes de ler o código):
//
//   shared[]  =  [ halo_esq (radius) | bloco (blockDim.x) | halo_dir (radius) ]
//                 índices 0..radius    radius..radius+bdim   radius+bdim..radius+bdim+radius
//
// Cada thread `tid` do bloco é dono do elemento em shared[tid + radius].
// Os primeiros `radius` threads (tid < radius) também carregam um elemento
// extra de halo à esquerda. Os últimos `radius` threads (tid >= bdim - radius)
// também carregam um elemento extra de halo à direita.
//
// Outra correção: os índices usam `blockDim.x` real (lido em runtime via
// intrínseco), não o `tile_size` fixado em tempo de geração do IR. Isso
// remove a dependência frágil de "quem lançar o kernel precisa usar
// exatamente blockDim.x == tile_size" — agora basta que blockDim.x seja
// menor ou igual ao tamanho alocado de memória compartilhada.
pub fn flir_to_llvm(module: &FlirModule, caps: &Option<DeviceCapabilities>) -> Result<String> {
    // --- 3.0 Ler a operação e os parâmetros do FLIR ---
    // `.first()` retorna Option<&FlirOp>. `.ok_or_else(...)` converte um
    // None em um Err com a mensagem dada — padrão comum em Rust para
    // transformar Option em Result antes de usar `?`.
    let op = module.ops.first().ok_or_else(|| anyhow!("FlirModule sem nenhuma operação"))?;
    let params = op.params.as_ref().ok_or_else(|| anyhow!("Operação '{}' sem params", op.op))?;

    let radius: i64 = params["radius"].as_i64().unwrap_or(1);
    let coeffs: Vec<f64> = params["coeffs"]
        .as_array()
        .ok_or_else(|| anyhow!("'coeffs' ausente ou não é array"))?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect();

    // Sanidade: o número de coeficientes tem que bater com 2*radius + 1
    // (um coeficiente por vizinho, incluindo o centro).
    if coeffs.len() as i64 != 2 * radius + 1 {
        return Err(anyhow!(
            "Esperado {} coeficientes para radius={}, recebido {}",
            2 * radius + 1,
            radius,
            coeffs.len()
        ));
    }

    // --- 3.1 Inicializar o LLVM para o target NVPTX ---
    // que inicializa o suporte ao target NATIVO da máquina host (ex: x86_64)
    // — isso não tem nenhum efeito sobre a capacidade do LLVM de gerar
    // código para NVPTX. Sem inicializar o NVPTX explicitamente, a busca
    // pelo target "nvptx64-nvidia-cuda" (feita depois, em compile_to_ptx)
    // falha em runtime.
    Target::initialize_nvptx(&InitializationConfig::default());

    let context = Context::create();
    let llvm_module = context.create_module("basalto_kernel");

    // Deixa explícito no módulo qual é o target — evita avisos de
    // incompatibilidade entre o triple do módulo e o do TargetMachine
    // usado depois em compile_to_ptx.
    llvm_module.set_triple(&TargetTriple::create("nvptx64-nvidia-cuda"));

    // --- 3.2 Tipos básicos que vamos usar ---
    let f64_type = context.f64_type();
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    let void_type = context.void_type();

    // --- 3.3 Assinatura do kernel: void basalto_kernel(double* x, double* y, int N) ---
    // Os ponteiros de parâmetro entram em address space genérico (0); dentro
    // da função, convertemos explicitamente para address space 1 (global),
    // que é o que o NVPTX espera para leitura/escrita em memória global.
    let generic_ptr = f64_type.ptr_type(AddressSpace(0));
    let fn_type = void_type.fn_type(&[generic_ptr.into(), generic_ptr.into(), i32_type.into()], false);
    let kernel_fn = llvm_module.add_function("basalto_kernel", fn_type, None);

    let x_ptr = kernel_fn.get_param(0).unwrap().into_pointer_value();
    let y_ptr = kernel_fn.get_param(1).unwrap().into_pointer_value();
    let n_param = kernel_fn.get_param(2).unwrap().into_int_value();

    let entry = context.append_basic_block(kernel_fn, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry);

    let x_global = builder.build_address_space_cast(x_ptr, f64_type.ptr_type(AddressSpace(1)), "x_global");
    let y_global = builder.build_address_space_cast(y_ptr, f64_type.ptr_type(AddressSpace(1)), "y_global");

    // --- 3.4 Declarar a memória compartilhada dinâmica ---
    // O tipo é [0 x double] (array de tamanho zero — o tamanho real é
    // definido em runtime, no momento do lançamento do kernel, não aqui).
    // O importante é que a variável em si — não um ponteiro para ela —
    // resida em AddressSpace(3) (memória compartilhada), passado como
    // segundo argumento de add_global, e NÃO embutido dentro do tipo.
    let shared_array_type = f64_type.array_type(0);
    let shared_global = llvm_module.add_global(shared_array_type, Some(AddressSpace(3)), "shared_mem");
    shared_global.set_linkage(inkwell::module::Linkage::External);
    shared_global.set_alignment(8); // double precisa de alinhamento de 8 bytes
    let base_ptr = shared_global.as_pointer_value();

    // --- 3.5 Intrínsecos NVPTX para ler threadIdx / blockIdx / blockDim ---
    let i32_fn_type = i32_type.fn_type(&[], false);
    let tid_fn = llvm_module.add_function("llvm.nvvm.read.ptx.sreg.tid.x", i32_fn_type, None);
    let bid_fn = llvm_module.add_function("llvm.nvvm.read.ptx.sreg.ctaid.x", i32_fn_type, None);
    let bdim_fn = llvm_module.add_function("llvm.nvvm.read.ptx.sreg.ntid.x", i32_fn_type, None);
    let barrier_fn = llvm_module.add_function("llvm.nvvm.barrier0", void_type.fn_type(&[], false), None);

    // `build_call(...)` retorna um CallSiteValue; `.try_as_basic_value()`
    // dá um Either<valor, void>; `.left()` pega o lado "tem valor" como
    // Option; `.unwrap()` assume que não é void (seguro aqui porque
    // sabemos que essas funções retornam i32).
    let tid = builder.build_call(tid_fn, &[], "tid").try_as_basic_value().left().unwrap().into_int_value();
    let bid = builder.build_call(bid_fn, &[], "bid").try_as_basic_value().left().unwrap().into_int_value();
    let bdim = builder.build_call(bdim_fn, &[], "bdim").try_as_basic_value().left().unwrap().into_int_value();

    // Convertemos tudo para i64 para fazer aritmética de índice sem
    // preocupação com overflow de 32 bits em malhas grandes.
    let tid64 = builder.build_int_cast(tid, i64_type, "tid64");
    let bid64 = builder.build_int_cast(bid, i64_type, "bid64");
    let bdim64 = builder.build_int_cast(bdim, i64_type, "bdim64");
    let n64 = builder.build_int_cast(n_param, i64_type, "n64");

    // Pequeno helper local (closure) para criar constantes i64 com menos
    // ruído visual — em Rust, closures que só leem variáveis do ambiente
    // (aqui, nada externo) podem ser `move` ou não; como não capturamos
    // nada, não precisa.
    let const_i64 = |v: i64| i64_type.const_int(v as u64, false);

    // global_idx = blockIdx.x * blockDim.x + threadIdx.x
    let tile_start = builder.build_int_mul(bid64, bdim64, "tile_start"); // primeiro índice global do bloco
    let global_idx = builder.build_int_add(tile_start, tid64, "global_idx");

    // --- 3.6 Bounds check do grid (thread processa elemento fora do array?) ---
    let cond_out_of_grid = builder.build_int_compare(IntPredicate::UGE, global_idx, n64, "cond_out_of_grid");
    let exit_block = context.append_basic_block(kernel_fn, "exit");
    let body_block = context.append_basic_block(kernel_fn, "body");
    builder.build_conditional_branch(cond_out_of_grid, exit_block, body_block);
    builder.position_at_end(body_block);

    // Helper para ler da memória global com clamp para zero fora dos
    // limites [0, N) — usado tanto para o elemento central quanto halo.
    // Definido como função Rust normal (não closure) porque precisa do
    // builder/tipos por referência; em Rust, funções aninhadas dentro de
    // outra função são permitidas e não capturam variáveis do escopo
    // externo automaticamente (diferente de closures).
    let safe_load = |idx: inkwell::values::IntValue,
                      builder: &inkwell::builder::Builder,
                      x_global: inkwell::values::PointerValue,
                      n64: inkwell::values::IntValue,
                      zero: inkwell::values::FloatValue|
     -> inkwell::values::FloatValue {
        let neg = builder.build_int_compare(IntPredicate::SLT, idx, const_i64(0), "neg");
        let ge_n = builder.build_int_compare(IntPredicate::UGE, idx, n64, "ge_n");
        let invalid = builder.build_or(neg, ge_n, "invalid"); // OR, não XOR
        let ptr = unsafe { builder.build_gep(x_global, &[idx], "ptr") };
        let loaded = builder.build_load(ptr, "loaded").into_float_value();
        builder.build_select(invalid, zero, loaded, "safe_val").into_float_value()
    };

    let zero = f64_type.const_float(0.0);

    // --- 3.7 Carregar elemento central: shared[tid + radius] = x[global_idx] ---
    let center_shared_idx = builder.build_int_add(tid64, const_i64(radius), "center_shared_idx");
    let center_val = safe_load(global_idx, &builder, x_global, n64, zero);
    let center_store_ptr = unsafe { builder.build_gep(base_ptr, &[const_i64(0), center_shared_idx], "center_store_ptr") };
    builder.build_store(center_store_ptr, center_val);

    // --- 3.8 Halo esquerdo: threads com tid < radius carregam um elemento extra ---
    // Índice global correto: tile_start - radius + tid  (SEM somar tid de novo)
    // Índice na shared: tid  (posições 0..radius-1)
    let left_cond = builder.build_int_compare(IntPredicate::SLT, tid64, const_i64(radius), "left_cond");
    let left_block = context.append_basic_block(kernel_fn, "left_halo");
    let after_left = context.append_basic_block(kernel_fn, "after_left");
    builder.build_conditional_branch(left_cond, left_block, after_left);
    builder.position_at_end(left_block);
    {
        let left_global_idx = builder.build_int_sub(
            builder.build_int_add(tile_start, tid64, "tile_start_plus_tid_tmp"), // == global_idx, recalculado localmente por clareza
            const_i64(radius),
            "left_global_idx",
        );
        // (equivalente a: tile_start + tid - radius)
        let left_val = safe_load(left_global_idx, &builder, x_global, n64, zero);
        let left_shared_idx = tid64; // 0..radius-1
        let left_store_ptr = unsafe { builder.build_gep(base_ptr, &[const_i64(0), left_shared_idx], "left_store_ptr") };
        builder.build_store(left_store_ptr, left_val);
    }
    builder.build_unconditional_branch(after_left);
    builder.position_at_end(after_left);

    // --- 3.9 Halo direito: threads com tid >= blockDim.x - radius carregam um elemento extra ---
    // Índice global correto: tile_start + blockDim.x + (tid - (blockDim.x - radius))
    // Índice na shared: radius + blockDim.x + (tid - (blockDim.x - radius))
    let right_threshold = builder.build_int_sub(bdim64, const_i64(radius), "right_threshold");
    let right_cond = builder.build_int_compare(IntPredicate::UGE, tid64, right_threshold, "right_cond");
    let right_block = context.append_basic_block(kernel_fn, "right_halo");
    let after_right = context.append_basic_block(kernel_fn, "after_right");
    builder.build_conditional_branch(right_cond, right_block, after_right);
    builder.position_at_end(right_block);
    {
        // offset dentro do halo direito: 0..radius-1
        let right_offset = builder.build_int_sub(tid64, right_threshold, "right_offset");
        let right_global_idx = builder.build_int_add(
            builder.build_int_add(tile_start, bdim64, "tile_start_plus_bdim"),
            right_offset,
            "right_global_idx",
        );
        let right_val = safe_load(right_global_idx, &builder, x_global, n64, zero);
        let right_shared_idx = builder.build_int_add(
            builder.build_int_add(const_i64(radius), bdim64, "radius_plus_bdim"),
            right_offset,
            "right_shared_idx",
        );
        let right_store_ptr = unsafe { builder.build_gep(base_ptr, &[const_i64(0), right_shared_idx], "right_store_ptr") };
        builder.build_store(right_store_ptr, right_val);
    }
    builder.build_unconditional_branch(after_right);
    builder.position_at_end(after_right);

    // --- 3.10 Sincronizar: todos os threads do bloco terminaram de carregar a tile ---
    builder.build_call(barrier_fn, &[], "sync_after_load");

    // --- 3.11 Calcular o stencil lendo da memória compartilhada ---
    // O total de elementos válidos na shared é: radius (halo esq) + blockDim.x + radius (halo dir)
    // Calculado em runtime a partir de bdim64 — não do tile_size fixado na
    // geração do IR, para não depender de blockDim.x == tile_size no lançamento.
    let total_shared_elems = builder.build_int_add(bdim64, const_i64(2 * radius), "total_shared_elems");

    let mut result = f64_type.const_float(0.0);
    // `coeffs.iter().enumerate()` dá (índice, &valor) para cada coeficiente.
    for (r, coeff) in coeffs.iter().enumerate() {
        // r vai de 0 até 2*radius; subtraindo radius, vira -radius..+radius
        let r_offset = (r as i64) - radius;
        let coeff_val = f64_type.const_float(*coeff);

        let neighbor_shared_idx = builder.build_int_add(center_shared_idx, const_i64(r_offset), "neighbor_shared_idx");

        // Esse bounds check é uma rede de segurança: matematicamente, para
        // qualquer r_offset dentro de [-radius, +radius], o índice deveria
        // sempre cair dentro de [0, total_shared_elems) — desde que os
        // passos 3.7 a 3.9 tenham preenchido a tile corretamente. Mantemos
        // a checagem mesmo assim para nunca ler memória fora dos limites
        // em caso de algum outro bug futuro.
        let valid_low = builder.build_int_compare(IntPredicate::SGE, neighbor_shared_idx, const_i64(0), "valid_low");
        let valid_high = builder.build_int_compare(IntPredicate::SLT, neighbor_shared_idx, total_shared_elems, "valid_high");
        let valid = builder.build_and(valid_low, valid_high, "valid");
        let safe_idx = builder.build_select(valid, neighbor_shared_idx, const_i64(0), "safe_idx").into_int_value();

        let neighbor_ptr = unsafe { builder.build_gep(base_ptr, &[const_i64(0), safe_idx], "neighbor_ptr") };
        let neighbor_val = builder.build_load(neighbor_ptr, "neighbor_val").into_float_value();
        let weighted = builder.build_float_mul(neighbor_val, coeff_val, "weighted");
        result = builder.build_float_add(result, weighted, "accum");
    }

    // --- 3.12 Segunda barreira (opcional, mas segura antes de reusar a shared em outra chamada) ---
    builder.build_call(barrier_fn, &[], "sync_after_compute");

    // --- 3.13 Escrever resultado: y[global_idx] = result ---
    let out_ptr = unsafe { builder.build_gep(y_global, &[global_idx], "out_ptr") };
    builder.build_store(out_ptr, result);
    builder.build_unconditional_branch(exit_block);

    // --- 3.14 Bloco de saída ---
    builder.position_at_end(exit_block);
    builder.build_return(None);

    // --- 3.15 Metadado NVPTX marcando a função como kernel (entry point) ---
    // Formato esperado: !{função, !"kernel", i32 1}
    let func_meta = kernel_fn.as_metadata_value();
    let kernel_str = context.metadata_string("kernel");
    let one_i32 = context.i32_type().const_int(1, false).as_metadata_value();
    let md_node = context.metadata_node(&[func_meta.into(), kernel_str.into(), one_i32.into()]);
    llvm_module.add_named_metadata("nvvm.annotations", &[md_node]);

    // caps ainda não influencia a geração de código além do tile_size usado
    // em build_flir (para dimensionar shared_mem_bytes). Registrar aqui só
    // para visibilidade em debug.
    if let Some(c) = caps {
        eprintln!(
            "[FLIR] IR gerado para GPU com {} SMs, max_threads_per_block={}",
            c.multi_processor_count, c.max_threads_per_block
        );
    }

    Ok(llvm_module.print_to_string().to_string())
}

// ============================================================================
// 4. COMPILAÇÃO: LLVM IR (texto) → PTX (bytes)
// ============================================================================
pub fn compile_to_ptx(llvm_ir: &str, caps: &Option<DeviceCapabilities>) -> Result<Vec<u8>> {
    // Precisa inicializar o NVPTX aqui também, já que esta função pode ser
    // chamada isoladamente (ex: em testes), sem passar por flir_to_llvm antes.
    Target::initialize_nvptx(&InitializationConfig::default());

    let target_triple = TargetTriple::create("nvptx64-nvidia-cuda");
    let target = Target::from_triple(&target_triple).map_err(|e| anyhow!("Target error: {:?}", e))?;

    let target_cpu = caps
        .as_ref()
        .map(|c| format!("sm_{}{}", c.compute_capability_major, c.compute_capability_minor))
        .unwrap_or_else(|| "sm_70".to_string());

    let target_machine = target
        .create_target_machine(
            &target_triple,
            &target_cpu,
            "", // features de CPU vazio — pode incluir "+ptx78" etc. se necessário
            inkwell::targets::CodeGenOptLevel::Aggressive,
            inkwell::targets::RelocMode::Default,
            inkwell::targets::CodeModel::Default,
        )
        .ok_or_else(|| anyhow!("Não foi possível criar TargetMachine para {}", target_cpu))?;

    let context = Context::create();
    let mem_buffer = MemoryBuffer::create_from_memory_range_copy(llvm_ir.as_bytes(), "basalto_ir");
    let module = context
        .create_module_from_ir(mem_buffer)
        .map_err(|e| anyhow!("Falha ao parsear/criar módulo a partir do IR: {}", e))?;

    let output_buffer = target_machine
        .write_to_memory_buffer(&module, FileType::Assembly)
        .map_err(|e| anyhow!("Falha ao emitir PTX: {}", e))?;

    Ok(output_buffer.as_slice().to_vec())
}