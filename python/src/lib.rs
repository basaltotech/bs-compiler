use pyo3::prelude::*;
use basalto_tree::interceptor::compile_from_fx_graph;

#[pymodule]
fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compile_from_fx_graph, m)?)?;
    Ok(())
}
