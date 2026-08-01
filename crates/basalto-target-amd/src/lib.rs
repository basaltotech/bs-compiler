pub mod codegen {
    use anyhow::Result;

    // Stub textual para HSACO (AMD)
    pub fn generate_hsaco_textual(_flir: &str, _arch: &str) -> Result<Vec<u8>> {
        anyhow::bail!("Backend AMD textual ainda não implementado");
    }
}

pub mod runtime {
    use anyhow::Result;
    pub fn execute_hip(_binary: &[u8]) -> Result<()> {
        anyhow::bail!("Runtime AMD (HIP) ainda não implementado");
    }
}