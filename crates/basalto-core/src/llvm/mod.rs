pub mod parser;
pub mod builder;
pub mod types;

use inkwell::context::Context;
use inkwell::module::Module;
use anyhow::Result;
use crate::flir_builder::FlirOp;

/// Função principal do módulo LLVM do Basalto.
/// Recebe o Contexto de fora para garantir que a memória do LLVM permaneça viva 
/// durante toda a etapa de Codegen e compilação do Target de hardware.
pub fn build_llvm_module<'ctx>(
    context: &'ctx Context, 
    ops: &[FlirOp],
    total_elements: u64,
) -> Result<Module<'ctx>> {
    // Cria o container do módulo atrelado ao ciclo de vida do contexto
    let module = context.create_module("basalto_kernel");
    
    // Chama a nossa função real e atualizada da Fase 1.3
    builder::build_agnostic_module(context, &module, ops, total_elements)?;
    
    Ok(module)
}
