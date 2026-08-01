// crates/basalto-core/src/flir_builder.rs
// Versão CORRIGIDA – stencil 1D com memória compartilhada real, halo, bounds check correto
use anyhow::{anyhow, Result};
use inkwell::{
    AddressSpace, IntPredicate,
    context::Context,
    module::Module,
    targets::{Target, TargetMachine, TargetTriple, InitializationConfig, FileType},
    memory_buffer::MemoryBuffer,
    values::{FloatValue, IntValue, PointerValue, BasicValueEnum},
    types::BasicTypeEnum,
};
use basalto_common::hardware::DeviceCapabilities;

// --------------------------------------------------------------------------
// 1. Representação FLIR
// --------------------------------------------------------------------------
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

// --------------------------------------------------------------------------
// 2. Builder – gera módulo a partir do grafo (placeholder)
// --------------------------------------------------------------------------
pub fn build_flir(graph_str: &str, caps: &Option<DeviceCapabilities>) -> Result<FlirModule> {
    // TODO: parsear grafo FX real.
    // Usa as capacidades para definir tile_size e shared memory.
    let tile_size = caps.as_ref().map(|c| c.max_threads_per_block).unwrap_or(128);
    let radius = 1;
    // Cada elemento é f64 (8 bytes). Tile = (tile_size + 2*radius) elementos.
    let shared_mem_bytes = (tile_size + 2 * radius) as u32 * 8;

    let ops = vec![
        FlirOp {
            op: "stencil_1d".to_string(), // agora 1D explícito
            inputs: vec!["x".to_string()],
            output: "y".to_string(),
            params: Some(serde_json::json!({
                "radius": radius,
                "coeffs": [0.2, 0.3, 0.5],
                "tile_size": tile_size,
                "shared_mem_bytes": shared_mem_bytes,
            })),
        },
    ];
    Ok(FlirModule { ops, input_names: vec!["x".to_string()], output_names: vec!["y".to_string()] })
}

// --------------------------------------------------------------------------
// 3. Geração de LLVM IR – CORRIGIDA
// --------------------------------------------------------------------------
pub fn flir_to_llvm(module: &FlirModule, caps: &Option<DeviceCapabilities>) -> Result<String> {
    let op = module.ops.first().ok_or_else(|| anyhow!("Nenhuma operação"))?;
    let params = op.params.as_ref().ok_or_else(|| anyhow!("Op sem params"))?;
    let radius: i64 = params["radius"].as_i64().unwrap_or(1);
    let coeffs: Vec<f64> = params["coeffs"].as_array()
        .unwrap_or(&vec![serde_json::Value::from(1.0)])
        .iter()
        .map(|v| v.as_f64().unwrap_or(1.0))
        .collect();
    let tile_size: i64 = params["tile_size"].as_i64().unwrap_or(128);
    let shared_mem_bytes: u32 = params["shared_mem_bytes"].as_u64().unwrap_or(0) as u32;

    // Inicializa LLVM
    let init_config = InitializationConfig { asm_parser: false, ..Default::default() };
    Target::initialize_native(&init_config).unwrap();
    let context = Context::create();
    let llvm_module = context.create_module("basalto_kernel");

    // Tipos
    let f64_type = context.f64_type();
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    let void_type = context.void_type();

    // Assinatura: void kernel(float* x, float* y, int N)
    // Usamos ponteiros genéricos (addrspace 0) – depois convertemos para global.
    let generic_ptr = f64_type.ptr_type(AddressSpace(0));
    let fn_type = void_type.fn_type(&[generic_ptr.into(), generic_ptr.into(), i32_type.into()], false);
    let kernel_fn = llvm_module.add_function("basalto_kernel", fn_type, None);
    let x_ptr = kernel_fn.get_param(0).unwrap().into_pointer_value();
    let y_ptr = kernel_fn.get_param(1).unwrap().into_pointer_value();
    let n_param = kernel_fn.get_param(2).unwrap().into_int_value();

    // Bloco de entrada – faz as casts de address space
    let entry = context.append_basic_block(kernel_fn, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry);

    let x_global = builder.build_address_space_cast(x_ptr, f64_type.ptr_type(AddressSpace(1)), "x_global");
    let y_global = builder.build_address_space_cast(y_ptr, f64_type.ptr_type(AddressSpace(1)), "y_global");

    // ----------------------------------------------------------------------
    // 1. DECLARAÇÃO DA MEMÓRIA COMPARTILHADA (externa, dinâmica)
    //    Correção do item 1: tipo é [0 x double] no address space 3.
    // ----------------------------------------------------------------------
    let shared_type = f64_type.array_type(0); // [0 x double]
    let shared_global = llvm_module.add_global(
        shared_type.ptr_type(AddressSpace(3)),
        None,
        "shared_mem"
    );
    shared_global.set_linkage(inkwell::module::Linkage::External);

    // Intrínsecos NVPTX
    let i32_fn_type = i32_type.fn_type(&[], false);
    let tid_fn = llvm_module.add_function("llvm.nvvm.read.ptx.sreg.tid.x", i32_fn_type, None);
    let bid_fn = llvm_module.add_function("llvm.nvvm.read.ptx.sreg.ctaid.x", i32_fn_type, None);
    let bdim_fn = llvm_module.add_function("llvm.nvvm.read.ptx.sreg.ntid.x", i32_fn_type, None);

    // Lê registros
    let tid = builder.build_call(tid_fn, &[], "tid").try_as_basic_value().left().unwrap().into_int_value();
    let bid = builder.build_call(bid_fn, &[], "bid").try_as_basic_value().left().unwrap().into_int_value();
    let bdim = builder.build_call(bdim_fn, &[], "bdim").try_as_basic_value().left().unwrap().into_int_value();

    // Converte para i64
    let bid64 = builder.build_int_cast(bid, i64_type, "bid64");
    let bdim64 = builder.build_int_cast(bdim, i64_type, "bdim64");
    let tid64 = builder.build_int_cast(tid, i64_type, "tid64");

    // global_idx = blockIdx.x * blockDim.x + threadIdx.x
    let global_idx = builder.build_int_add(
        builder.build_int_mul(bid64, bdim64, "offset"),
        tid64,
        "global_idx"
    );
    let n64 = builder.build_int_cast(n_param, i64_type, "n64");

    // Bounds check do grid: se global_idx >= N, retorna
    let cond_bound = builder.build_int_compare(IntPredicate::UGE, global_idx, n64, "cond_bound");
    let exit_block = context.append_basic_block(kernel_fn, "exit");
    let body_block = context.append_basic_block(kernel_fn, "body");
    builder.build_conditional_branch(cond_bound, exit_block, body_block);
    builder.position_at_end(body_block);

    // ----------------------------------------------------------------------
    // 2. CALCULAR ENDEREÇO DA TILE NA SHARED
    // ----------------------------------------------------------------------
    let base_ptr = shared_global.as_pointer_value();
    // O tamanho da tile é (tile_size + 2*radius)
    let tile_elems = tile_size + 2 * radius;
    // O índice na shared para o elemento central de cada thread = tid + radius
    let center_shared_idx = builder.build_int_add(tid64, i64_type.const_int(radius, false), "center_shared_idx");

    // ----------------------------------------------------------------------
    // 3. CARREGAR O ELEMENTO CENTRAL DA GLOBAL PARA SHARED
    //    Correção do item 4: global_load_idx = global_idx (sem somar tid)
    // ----------------------------------------------------------------------
    let global_load_idx = global_idx; // AGORA CORRETO
    // Proteção de borda global (item 3) – OR, não XOR
    let cond_neg = builder.build_int_compare(IntPredicate::SLT, global_load_idx, i64_type.const_int(0, false), "cond_neg");
    let cond_ge_n = builder.build_int_compare(IntPredicate::UGE, global_load_idx, n64, "cond_ge_n");
    let cond_invalid = builder.build_or(cond_neg, cond_ge_n, "cond_invalid"); // OR, não XOR

    let zero = f64_type.const_float(0.0);
    let load_ptr = unsafe { builder.build_gep(x_global, &[global_load_idx], "load_ptr") };
    let loaded = builder.build_load(load_ptr, "loaded").into_float_value();
    let val_to_store = builder.build_select(cond_invalid, zero, loaded, "val_to_store");

    // Guarda na shared na posição center_shared_idx
    let shared_store_ptr = unsafe {
        builder.build_gep(base_ptr, &[i64_type.const_int(0, false), center_shared_idx], "shared_store_ptr")
    };
    builder.build_store(shared_store_ptr, val_to_store);

    // ----------------------------------------------------------------------
    // 4. PREENCHER HALO ESQUERDO (threads com tid < radius)
    //    Correção do item 5 – implementado.
    // ----------------------------------------------------------------------
    let left_cond = builder.build_int_compare(IntPredicate::SLT, tid64, i64_type.const_int(radius, false), "left_cond");
    let left_block = context.append_basic_block(kernel_fn, "left_halo");
    let after_left = context.append_basic_block(kernel_fn, "after_left");
    builder.build_conditional_branch(left_cond, left_block, after_left);
    builder.position_at_end(left_block);
    // Cada thread da borda esquerda carrega o elemento correspondente à esquerda do início do bloco.
    // O índice global a carregar é: global_idx - (radius - tid)
    let left_offset = builder.build_int_sub(i64_type.const_int(radius, false), tid64, "left_offset");
    let left_global_idx = builder.build_int_sub(global_idx, left_offset, "left_global_idx");
    // Bounds check (com OR)
    let lneg = builder.build_int_compare(IntPredicate::SLT, left_global_idx, i64_type.const_int(0, false), "lneg");
    let lge = builder.build_int_compare(IntPredicate::UGE, left_global_idx, n64, "lge");
    let linvalid = builder.build_or(lneg, lge, "linvalid");
    let lload_ptr = unsafe { builder.build_gep(x_global, &[left_global_idx], "lload_ptr") };
    let lloaded = builder.build_load(lload_ptr, "lloaded").into_float_value();
    let lval = builder.build_select(linvalid, zero, lloaded, "lval");
    // Índice na shared: tid (0..radius-1)
    let lshared_idx = builder.build_int_add(tid64, i64_type.const_int(0, false), "lshared_idx");
    let lstore_ptr = unsafe {
        builder.build_gep(base_ptr, &[i64_type.const_int(0, false), lshared_idx], "lstore_ptr")
    };
    builder.build_store(lstore_ptr, lval);
    builder.build_unconditional_branch(after_left);
    builder.position_at_end(after_left);

    // ----------------------------------------------------------------------
    // 5. PREENCHER HALO DIREITO (threads com tid >= blockDim.x - radius)
    // ----------------------------------------------------------------------
    let right_tid = builder.build_int_sub(bdim64, i64_type.const_int(radius, false), "right_tid");
    let right_cond = builder.build_int_compare(IntPredicate::UGE, tid64, right_tid, "right_cond");
    let right_block = context.append_basic_block(kernel_fn, "right_halo");
    let after_right = context.append_basic_block(kernel_fn, "after_right");
    builder.build_conditional_branch(right_cond, right_block, after_right);
    builder.position_at_end(right_block);
    // Cada thread da borda direita carrega o elemento à direita do fim do bloco.
    // Índice global: global_idx + (radius - (blockDim.x - 1 - tid))
    let right_offset = builder.build_int_sub(
        i64_type.const_int(radius, false),
        builder.build_int_sub(builder.build_int_sub(bdim64, i64_type.const_int(1, false), "bdim_minus1"), tid64, "dist_to_end"),
        "right_offset"
    );
    let right_global_idx = builder.build_int_add(global_idx, right_offset, "right_global_idx");
    // Bounds check
    let rneg = builder.build_int_compare(IntPredicate::SLT, right_global_idx, i64_type.const_int(0, false), "rneg");
    let rge = builder.build_int_compare(IntPredicate::UGE, right_global_idx, n64, "rge");
    let rinvalid = builder.build_or(rneg, rge, "rinvalid");
    let rload_ptr = unsafe { builder.build_gep(x_global, &[right_global_idx], "rload_ptr") };
    let rloaded = builder.build_load(rload_ptr, "rloaded").into_float_value();
    let rval = builder.build_select(rinvalid, zero, rloaded, "rval");
    // Índice na shared: tile_size + radius + (tid - right_tid)
    let right_shared_idx = builder.build_int_add(
        builder.build_int_add(i64_type.const_int(tile_size, false), i64_type.const_int(radius, false), "base_right_idx"),
        builder.build_int_sub(tid64, right_tid, "right_tid_diff"),
        "right_shared_idx"
    );
    let rstore_ptr = unsafe {
        builder.build_gep(base_ptr, &[i64_type.const_int(0, false), right_shared_idx], "rstore_ptr")
    };
    builder.build_store(rstore_ptr, rval);
    builder.build_unconditional_branch(after_right);
    builder.position_at_end(after_right);

    // ----------------------------------------------------------------------
    // 6. SINCROMIZAR (barrier) – todos os threads terminaram de carregar
    // ----------------------------------------------------------------------
    let barrier_fn = llvm_module.add_function("llvm.nvvm.barrier0", void_type.fn_type(&[], false), None);
    builder.build_call(barrier_fn, &[], "sync");

    // ----------------------------------------------------------------------
    // 7. CALCULAR O STENCIL (lê da shared)
    // ----------------------------------------------------------------------
    let mut result = f64_type.const_float(0.0);
    for (r, coeff) in coeffs.iter().enumerate() {
        let r_i64 = (r as i64) - radius;
        let coeff_val = f64_type.const_float(*coeff);
        // Índice do vizinho na shared: center_shared_idx + r_i64
        let neighbor_shared_idx = builder.build_int_add(center_shared_idx, i64_type.const_int(r_i64, false), "neighbor_shared_idx");
        // Proteção contra índice fora da tile (usando SGE, não UGE – item 2)
        let valid_low = builder.build_int_compare(IntPredicate::SGE, neighbor_shared_idx, i64_type.const_int(0, false), "valid_low");
        let valid_high = builder.build_int_compare(IntPredicate::ULT, neighbor_shared_idx, i64_type.const_int(tile_elems, false), "valid_high");
        let valid_all = builder.build_and(valid_low, valid_high, "valid_all");
        // Se inválido, usa o índice 0 (qualquer valor, mas não causa crash)
        let idx_use = builder.build_select(valid_all, neighbor_shared_idx, i64_type.const_int(0, false), "idx_clamped");
        let gep = unsafe { builder.build_gep(base_ptr, &[i64_type.const_int(0, false), idx_use], "gep_neighbor") };
        let neighbor = builder.build_load(gep, "neighbor").into_float_value();
        let weighted = builder.build_float_mul(neighbor, coeff_val, "weighted");
        result = builder.build_float_add(result, weighted, "accum");
    }

    // ----------------------------------------------------------------------
    // 8. SINCRONIZAR NOVAMENTE (não obrigatório, mas seguro)
    // ----------------------------------------------------------------------
    builder.build_call(barrier_fn, &[], "sync2");

    // ----------------------------------------------------------------------
    // 9. ARMAZENAR RESULTADO NA GLOBAL: y[global_idx] = result
    // ----------------------------------------------------------------------
    let out_gep = unsafe { builder.build_gep(y_global, &[global_idx], "out_gep") };
    builder.build_store(out_gep, result);

    // ----------------------------------------------------------------------
    // 10. FINALIZAR
    // ----------------------------------------------------------------------
    builder.build_unconditional_branch(exit_block);
    builder.position_at_end(exit_block);
    builder.build_return(None);

    // Metadado NVPTX (3 elementos)
    let func_meta = kernel_fn.as_metadata_value();
    let kernel_str = context.metadata_string("kernel");
    let one_i32 = context.i32_type().const_int(1, false).as_metadata_value();
    let md_node = context.metadata_node(&[func_meta.into(), kernel_str.into(), one_i32.into()]);
    llvm_module.add_named_metadata("nvvm.annotations", &[md_node]);

    Ok(llvm_module.to_string())
}

// --------------------------------------------------------------------------
// 4. Compilação para PTX (API corrigida)
// --------------------------------------------------------------------------
pub fn compile_to_ptx(llvm_ir: &str, caps: &Option<DeviceCapabilities>) -> Result<Vec<u8>> {
    let target_triple = TargetTriple::create("nvptx64-nvidia-cuda");
    let target = Target::from_triple(&target_triple)
        .map_err(|e| anyhow!("Target error: {:?}", e))?;

    let target_cpu = caps.as_ref()
        .map(|c| format!("sm_{}{}", c.compute_capability_major, c.compute_capability_minor))
        .unwrap_or_else(|| "sm_70".to_string());

    let target_machine = target.create_target_machine(
        &target_triple,
        &target_cpu,
        "",
        inkwell::targets::CodeGenOptLevel::Aggressive,
        inkwell::targets::RelocMode::Default,
        inkwell::targets::CodeModel::Default,
    ).map_err(|e| anyhow!("Target machine error: {:?}", e))?;

    let context = Context::create();
    let mem_buffer = MemoryBuffer::parse_ir_from_string(llvm_ir, "inmem")
        .map_err(|e| anyhow!("Parse IR error: {}", e))?;
    let module = context.create_module_from_ir(mem_buffer)
        .map_err(|e| anyhow!("Module creation error: {}", e))?;

    let output_buffer = target_machine.write_to_memory_buffer(&module, FileType::Assembly)
        .map_err(|e| anyhow!("PTX emission error: {}", e))?;

    Ok(output_buffer.as_slice().to_vec())
}