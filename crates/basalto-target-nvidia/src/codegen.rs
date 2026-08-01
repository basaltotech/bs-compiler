use anyhow::{anyhow, Result};

// ============================================================
// BACKEND 1: GERADOR TEXTUAL (SEMPRE DISPONÍVEL)
// ============================================================

/// Gera PTX textual diretamente a partir da FLIR (protótipo rápido).
pub fn generate_ptx(flir: &str, arch: &str) -> Result<Vec<u8>> {
    if flir.is_empty() {
        return Err(anyhow!("FLIR está vazio. Não há código para gerar."));
    }

    let target_arch = if arch.starts_with("sm_") { arch } else { "sm_80" };

    let mut ptx_code = String::new();
    ptx_code.push_str(".version 7.5\n");
    ptx_code.push_str(&format!(".target {}\n", target_arch));
    ptx_code.push_str(".address_size 64\n\n");

    ptx_code.push_str(".entry basalto_kernel (\n");
    ptx_code.push_str("    .param .u64 input_ptr,\n");
    ptx_code.push_str("    .param .u64 output_ptr\n");
    ptx_code.push_str(")\n{\n");

    ptx_code.push_str("    .reg .b64 %rd<4>;\n");
    ptx_code.push_str("    .reg .f32 %f<2>;\n");
    ptx_code.push_str("    .reg .b32 %r<1>;\n\n");

    ptx_code.push_str("    mov.u32 %r0, %tid.x;\n");
    ptx_code.push_str("    ld.param.u64 %rd1, [input_ptr];\n");
    ptx_code.push_str("    ld.param.u64 %rd2, [output_ptr];\n");
    ptx_code.push_str("    ld.global.f32 %f0, [%rd1];\n");
    ptx_code.push_str("    st.global.f32 [%rd2], %f0;\n");
    ptx_code.push_str("    ret;\n");
    ptx_code.push_str("}\n");

    Ok(ptx_code.into_bytes())
}

// ============================================================
// BACKEND 2: GERADOR LLVM (OTIMIZADO, FEATURE-GATED)
// ============================================================

#[cfg(feature = "llvm-codegen")]
pub mod llvm {
    use super::*;
    use inkwell::module::Module;
    use inkwell::targets::{CodeGenFileType, Target, TargetMachine, TargetOptions};

    /// Compila um módulo LLVM (já otimizado) para PTX.
    pub fn compile_module_to_ptx(module: &Module, arch: &str) -> Result<Vec<u8>> {
        Target::initialize_nvptx(&TargetOptions::default());

        let target_triple = "nvptx64-nvidia-cuda";
        let cpu = arch;
        let features = "";

        let target = Target::from_triple(target_triple)
            .map_err(|e| anyhow!("Falha ao criar target NVPTX: {}", e))?;

        let target_machine = target
            .create_target_machine(
                &target_triple,
                cpu,
                features,
                inkwell::OptimizationLevel::Aggressive,
                &TargetOptions::default(),
            )
            .ok_or_else(|| anyhow!("Falha ao criar TargetMachine para arch: {}", arch))?;

        let buffer = target_machine
            .write_to_memory_buffer(module, CodeGenFileType::Assembly)
            .map_err(|e| anyhow!("Falha ao escrever PTX: {}", e))?;

        let ptx_bytes = buffer.as_slice().to_vec();

        if ptx_bytes.is_empty() {
            return Err(anyhow!("PTX gerado está vazio"));
        }

        Ok(ptx_bytes)
    }
}