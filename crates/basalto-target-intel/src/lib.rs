pub mod codegen {
    use anyhow::Result;

    pub fn generate_spirv_textual(_flir: &str, _arch: &str) -> Result<Vec<u8>> {
        anyhow::bail!("Backend Intel (SPIR-V) textual ainda não implementado");
    }
}

pub mod runtime {
    use anyhow::Result;
    pub fn execute_level_zero(_binary: &[u8]) -> Result<()> {
        anyhow::bail!("Runtime Intel (Level Zero) ainda não implementado");
    }
}