pub mod interceptor;
pub mod local_cache;
pub mod cluster_cache;
pub mod executor;

use pyo3::prelude::*;
use pyo3::types::PyString;
use anyhow::Result;

/// Função chamada pelo Python que recebe o grafo do PyTorch FX.
/// Usamos `&Bound<'_, PyAny>` (sintaxe moderna do PyO3 >= 0.21) para manipular o objeto Python.
#[pyfunction]
fn compile_from_fx_graph(fx_graph: &Bound<'_, PyAny>) -> PyResult<u64> {
    // 1. Extrai a representação em string do grafo do PyTorch
    // Em Python seria o equivalente a chamar: str(fx_graph) ou fx_graph.code
    let graph_str: String = fx_graph.str()?.extract()?;

    // 2. Extrai os shapes (metadados dos tensores) enviados pelo PyTorch
    // Placeholder: Na prática, você varre os nós do FX Graph buscando o atributo 'shape'
    let shapes: Vec<Vec<usize>> = vec![vec![1, 3, 224, 224]]; 

    // 3. Executa o pipeline otimizado do Basalto
    let token = pipeline_principal(&graph_str, &shapes).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Erro no Basalto: {}", e))
    })?;

    // Retorna o token U64 do BLAKE3 direto para o Python
    Ok(token)
}

/// Orquestra a lógica enxuta do compilador que desenvolvemos
fn pipeline_principal(graph_str: &str, shapes: &[Vec<usize>]) -> Result<u64> {
    // a) Detecta GPU (Usa a versão enxuta com root via sysfs)
    let gpu = executor::detect_gpu()?; 
    
    // b) Otimiza o layout dos tensores
    let optimized_shapes = executor::reorganize_tensors(shapes)?;
    
    // c) Gera o Hash Único via BLAKE3 (Seguro e rápido para cache)
    let token = local_cache::hash_kernel(
        graph_str, 
        &optimized_shapes, 
        &gpu.vendor, 
        &gpu.arch, 
        &gpu.driver_version
    );
    
    // d) Se não estiver no cache, gera o PTX
    // let ptx = executor::generate_ptx(graph_str, &gpu.arch)?;
    
    Ok(token)
}

/// Módulo Python exposto via PyO3.
#[pymodule]
fn basalto_tree(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile_from_fx_graph, m)?)?;
    Ok(())
}
