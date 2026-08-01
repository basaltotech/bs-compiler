// crates/basalto-target-nvidia/src/codegen.rs
// Este módulo é um placeholder; a geração de PTX é feita pelo `basalto-core/flir_builder`.
// Mantido para compatibilidade futura.

use anyhow::Result;

/// Gera PTX a partir de LLVM IR. (Não implementado diretamente aqui; use flir_builder::compile_to_ptx).
pub fn generate_ptx(_llvm_ir: &str) -> Result<Vec<u8>> {
    Err(anyhow::anyhow!(
        "Use basalto_core::flir_builder::compile_to_ptx para gerar PTX a partir de LLVM IR."
    ))
}