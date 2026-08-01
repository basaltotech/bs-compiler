use crate::{local_cache, cluster_cache, executor};
use basalto_gems::stride_view;
use basalto_core::{flir_builder, hasher};
use basalto_common::permissions::ensure_root_or_die;
use pyo3::prelude::*;
use anyhow::Result;

#[pyfunction]
pub fn compile_from_fx_graph(
    graph_str: String,
    shapes: Vec<Vec<usize>>,
    vendor: String,
    arch: String,
    driver_ver: String,
) -> PyResult<Vec<u8>> {
    // 1. Root obrigatório
    ensure_root_or_die().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // 2. Gems
    let optimized_shapes = stride_view::reorganize_tensors(&shapes)?;

    // 3. Hash
    let kernel_hash = hasher::hash_kernel(&graph_str, &optimized_shapes, &vendor, &arch, &driver_ver);

    // 4. Cache L1
    if let Some(binary) = local_cache::get(kernel_hash) {
        return Ok(binary);
    }

    // 5. Cache L2
    if let Some(binary) = cluster_cache::get(kernel_hash) {
        local_cache::put(kernel_hash, &binary);
        return Ok(binary);
    }

    // ============================================================
    // 6. Compilação (escolha do backend)
    // ============================================================

    // Decisão do backend via variável de ambiente
    #[cfg(feature = "llvm-codegen")]
    let use_llvm = std::env::var("BASALTO_CODEGEN")
        .map(|s| s == "llvm")
        .unwrap_or(true); // Padrão: LLVM se a feature estiver ativa

    #[cfg(not(feature = "llvm-codegen"))]
    let use_llvm = false; // Sempre textual

    let binary = if use_llvm {
        // ---------- CAMINHO LLVM (OTIMIZADO) ----------
        #[cfg(feature = "llvm-codegen")]
        {
            let flir_ops = flir_builder::build_flir(&graph_str)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let total_elements: u64 = optimized_shapes.iter()
                .fold(1, |acc, s| acc * s.iter().product::<usize>());

            let (ctx, module) = basalto_core::llvm::build_llvm_module(&flir_ops, total_elements)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            // Compila para PTX via LLVM
            match vendor.as_str() {
                "nvidia" => {
                    basalto_target_nvidia::llvm::compile_module_to_ptx(&module, &arch)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                }
                "amd" => {
                    // Futuro: basalto_target_amd::llvm::compile_module_to_hsaco(&module, &arch)?
                    Vec::new()
                }
                "intel" => {
                    // Futuro: basalto_target_intel::llvm::compile_module_to_spirv(&module, &arch)?
                    Vec::new()
                }
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Vendor não suportado")),
            }
        }

        #[cfg(not(feature = "llvm-codegen"))]
        {
            // Fallback seguro (nunca deve acontecer)
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("LLVM não compilado (feature desativada)"));
        }
    } else {
        // ---------- CAMINHO TEXTUAL (PROTÓTIPO) ----------
        let flir_str = flir_builder::build_flir_string(&graph_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        match vendor.as_str() {
            "nvidia" => {
                basalto_target_nvidia::codegen::generate_ptx(&flir_str, &arch)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            }
            "amd" => {
                // Futuro: generate_hsaco_textual
                Vec::new()
            }
            "intel" => {
                // Futuro: generate_spirv_textual
                Vec::new()
            }
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Vendor não suportado")),
        }
    };

    if binary.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Binário compilado vazio"));
    }

    // 7. Caches
    local_cache::put(kernel_hash, &binary);
    cluster_cache::put(kernel_hash, &binary);

    // 8. Executor
    executor::dispatch(&binary, &optimized_shapes)?;

    Ok(binary)
}