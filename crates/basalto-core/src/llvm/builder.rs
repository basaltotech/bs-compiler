use anyhow::{anyhow, Result};
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::builder::Builder;
use inkwell::values::{BasicValueEnum, PointerValue, IntValue};
use inkwell::types::BasicType;
use inkwell::AddressSpace;
use inkwell::passes::PassManager; // ⬅️ Import necessário
use crate::flir_builder::FlirOp;

/// Compila a lista de instruções SSA da FLIR para um módulo LLVM IR puramente agnóstico.
pub fn build_agnostic_module(
    ctx: &Context,
    module: &Module,
    ops: &[FlirOp],
    total_elements: u64,
) -> Result<()> {
    let builder = ctx.create_builder();
    
    // 1. Definição de Tipos e Tabela de Registradores SSA Virtuais
    let f32_type = ctx.f32_type();
    let i32_type = ctx.i32_type();
    let ptr_type = f32_type.ptr_type(AddressSpace::Generic);
    
    // Encontra o maior ID para dimensionar os vetores
    let max_id = ops.iter().map(|op| match op {
        FlirOp::DeclareArgument { id, .. } => *id,
        FlirOp::LoadCoordinate { id, .. } => *id,
        FlirOp::MathAdd { id, .. } => *id,
        FlirOp::MathMul { id, .. } => *id,
        FlirOp::StoreCoordinate { .. } => 0,
    }).max().unwrap_or(0);
    
    let mut ssa_registers: Vec<Option<BasicValueEnum>> = vec![None; max_id + 1];
    let mut argument_pointers: Vec<Option<PointerValue>> = vec![None; max_id + 1];

    // 2. Extrai e mapeia os argumentos globais da função (Ponteiros brutos de Tensores)
    let mut arg_types = Vec::new();
    for op in ops {
        if let FlirOp::DeclareArgument { .. } = op {
            arg_types.push(ptr_type.into());
        }
    }
    // Injeta o offset global do ID da thread/rank MPI como último parâmetro
    arg_types.push(i32_type.into());

    let fn_type = ctx.void_type().fn_type(&arg_types, false);
    let fn_val = module.add_function("basalto_agnostic_stencil", fn_type, None);

    // 3. Estruturação dos Blocos do Loop Espacial 
    let entry_block = ctx.append_basic_block(fn_val, "entry");
    let loop_cond_block = ctx.append_basic_block(fn_val, "loop_cond");
    let loop_body_block = ctx.append_basic_block(fn_val, "loop_body");
    let loop_end_block = ctx.append_basic_block(fn_val, "loop_end");

    // --- BLOCO: ENTRY ---
    builder.position_at_end(entry_block);
    let params = fn_val.get_params();
    
    let mut param_idx = 0;
    for op in ops {
        if let FlirOp::DeclareArgument { id, .. } = op {
            argument_pointers[*id] = Some(params[param_idx].into_pointer_value());
            param_idx += 1;
        }
    }
    let id_offset = params[param_idx].into_int_value();

    // Loop Index
    let i_var = builder.build_alloca(i32_type, "mesh_index")
        .map_err(|e| anyhow!("Falha alloca índice: {:?}", e))?;
    builder.build_store(i_var, id_offset)
        .map_err(|e| anyhow!("Falha inicialização índice: {:?}", e))?;
    builder.build_unconditional_branch(loop_cond_block)
        .map_err(|e| anyhow!("Falha branch inicial: {:?}", e))?;

    // --- BLOCO: LOOP COND ---
    builder.position_at_end(loop_cond_block);
    let current_i = builder.build_load(i32_type, i_var, "current_index")
        .map_err(|e| anyhow!("Falha load índice: {:?}", e))?.into_int_value();
    
    let total_elements_val = i32_type.const_int(total_elements, false);
    let is_active = builder.build_int_compare(inkwell::IntPredicate::ULT, current_i, total_elements_val, "is_active")
        .map_err(|e| anyhow!("Falha comparação loop: {:?}", e))?;
    
    builder.build_conditional_branch(is_active, loop_body_block, loop_end_block)
        .map_err(|e| anyhow!("Falha branch condicional: {:?}", e))?;

    // --- BLOCO: LOOP BODY (A varredura SSA real das operações FLIR) ---
    builder.position_at_end(loop_body_block);

    for op in ops {
        match op {
            FlirOp::DeclareArgument { .. } => {} // Já tratados no Entry
            
            FlirOp::LoadCoordinate { id, tensor_id, offsets } => {
                let tensor_ptr = argument_pointers[*tensor_id]
                    .ok_or_else(|| anyhow!("Uso de ponteiro de argumento não declarado: ID {}", tensor_id))?;
                
                let mut index_with_offset = current_i;
                if let Some(&first_offset) = offsets.first() {
                    if first_offset != 0 {
                        let offset_val = i32_type.const_int(first_offset as u64, true);
                        index_with_offset = builder.build_int_add(current_i, offset_val, "idx_offset")
                            .map_err(|e| anyhow!("Falha ao aplicar offset linear: {:?}", e))?;
                    }
                }

                let gep = unsafe { builder.build_gep(f32_type, tensor_ptr, &[index_with_offset], "gep")
                    .map_err(|e| anyhow!("Falha GEP na FLIR Op: {:?}", e))? };
                let val = builder.build_load(f32_type, gep, "loaded_val")
                    .map_err(|e| anyhow!("Falha Load na FLIR Op: {:?}", e))?;
                
                ssa_registers[*id] = Some(val);
            }
            
            FlirOp::MathAdd { id, lhs, rhs } => {
                let left = ssa_registers[*lhs].ok_or_else(|| anyhow!("SSA Reg não inicializado: %{}", lhs))?.into_float_value();
                let right = ssa_registers[*rhs].ok_or_else(|| anyhow!("SSA Reg não inicializado: %{}", rhs))?.into_float_value();
                
                let res = builder.build_float_add(left, right, "add_tmp")
                    .map_err(|e| anyhow!("Falha Add na FLIR Op: {:?}", e))?;
                ssa_registers[*id] = Some(res.into());
            }
            
            FlirOp::MathMul { id, lhs, rhs } => {
                let left = ssa_registers[*lhs].ok_or_else(|| anyhow!("SSA Reg não inicializado: %{}", lhs))?.into_float_value();
                let right = ssa_registers[*rhs].ok_or_else(|| anyhow!("SSA Reg não inicializado: %{}", rhs))?.into_float_value();
                
                let res = builder.build_float_mul(left, right, "mul_tmp")
                    .map_err(|e| anyhow!("Falha Mul na FLIR Op: {:?}", e))?;
                ssa_registers[*id] = Some(res.into());
            }
            
            FlirOp::StoreCoordinate { tensor_id, value_to_store } => {
                let dest_ptr = argument_pointers[*tensor_id]
                    .ok_or_else(|| anyhow!("Uso de destino de escrita não declarado: ID {}", tensor_id))?;
                let val = ssa_registers[*value_to_store]
                    .ok_or_else(|| anyhow!("Tentativa de gravar reg SSA vazio: %{}", value_to_store))?;
                
                let gep = unsafe { builder.build_gep(f32_type, dest_ptr, &[current_i], "store_gep")
                    .map_err(|e| anyhow!("Falha GEP Store: {:?}", e))? };
                builder.build_store(gep, val).map_err(|e| anyhow!("Falha Store de volta na malha: {:?}", e))?;
            }
        }
    }

    // Incrementa e fecha o bloco de repetição
    let step_one = i32_type.const_int(1, false);
    let next_i = builder.build_int_add(current_i, step_one, "next_index")
        .map_err(|e| anyhow!("Falha incremento índice: {:?}", e))?;
    builder.build_store(i_var, next_i)
        .map_err(|e| anyhow!("Falha store próximo índice: {:?}", e))?;
    builder.build_unconditional_branch(loop_cond_block)
        .map_err(|e| anyhow!("Falha retorno ao cabeçalho do loop: {:?}", e))?;

    // --- BLOCO: LOOP END ---
    builder.position_at_end(loop_end_block);
    builder.build_return(None).map_err(|e| anyhow!("Falha retorno do kernel: {:?}", e))?;

    // 4. VALIDAÇÃO ESTRUTURAL DA IR
    if let Err(e) = module.verify() {
        return Err(anyhow!("Verificação de Consistência Matemática do LLVM falhou: {}", e.to_string()));
    }

    // 5. ⚡ OTIMIZAÇÕES LLVM (APÓS VERIFICAÇÃO BEM-SUCEDIDA)
    //    Aplica passes de otimização padrão para melhorar o código gerado.
    let pass_manager = PassManager::create(module);
    pass_manager.add_instruction_combining_pass();   // Funde instruções redundantes
    pass_manager.add_reassociate_pass();             // Reassocia expressões aritméticas
    pass_manager.add_gvn_pass();                     // Eliminação de valores globais
    pass_manager.add_mem2reg_pass();                 // Promove variáveis de memória para registradores
    pass_manager.run_on(module);

    Ok(())
}