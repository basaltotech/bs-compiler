use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlirOp {
    pub op: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub params: Option<serde_json::Value>,
}

/// Converte o grafo para uma lista de operações FLIR (usado pelo LLVM).
pub fn build_flir(graph_str: &str) -> Result<Vec<FlirOp>> {
    // Placeholder: gera uma soma vetorial simples
    let ops = vec![
        FlirOp {
            op: "add".to_string(),
            inputs: vec!["A".to_string(), "B".to_string()],
            output: "C".to_string(),
            params: None,
        },
    ];
    Ok(ops)
}

/// Converte o grafo para uma string FLIR (usado pelo backend textual).
pub fn build_flir_string(graph_str: &str) -> Result<String> {
    // Placeholder: devolve o próprio grafo como string
    Ok(graph_str.to_string())
}