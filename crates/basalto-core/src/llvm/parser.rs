use anyhow::Result;
use crate::flir_builder::FlirOp;

/// Parseia uma string FLIR (formato JSON) para uma lista de operações.
/// Este é um placeholder. Em produção, você usaria serde_json diretamente.
pub fn parse_flir(json_str: &str) -> Result<Vec<FlirOp>> {
    let ops: Vec<FlirOp> = serde_json::from_str(json_str)?;
    Ok(ops)
}