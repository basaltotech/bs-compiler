// crates/basalto-tree/src/interceptor.rs
use anyhow::{anyhow, Result};
use std::ffi::c_void;
use std::sync::Arc;
use tokio::sync::mpsc;
use serde_json::Value;

use basalto_common::hardware::GpuIdentity;
use basalto_core::hasher::KernelMetadata;
use basalto_core::flir_builder::{build_flir, flir_to_llvm, compile_to_ptx};
use basalto_tree::local_cache::{self, LocalCache}; // agora importamos o módulo
use basalto_tree::executor::{execute_flir_kernel, KernelExecutionReport};

// O cluster_cache ainda não existe – deixamos comentado
// use basalto_tree::cluster_cache::ClusterCache;

// --------------------------------------------------------------------------
// Interceptor – orquestra todo o fluxo
// --------------------------------------------------------------------------
pub struct BasaltoInterceptor {
    local_cache: LocalCache,
    jit_sender: Option<mpsc::Sender<KernelExecutionReport>>,
}

impl BasaltoInterceptor {
    pub fn new(jit_sender: Option<mpsc::Sender<KernelExecutionReport>>) -> Self {
        Self {
            local_cache: LocalCache::new_with_capacity(10_000), // LRU com 10k entradas
            jit_sender,
        }
    }

    /// Ponto de entrada principal.
    pub fn compile_and_execute(
        &self,
        op: String,
        dtype: String,
        shape: Vec<usize>,
        job_id: Option<String>,
        device_ptr_x: *mut c_void,
        device_ptr_y: *mut c_void,
        n: i32,
    ) -> Result<()> {
        // --- 1. Validações iniciais ---
        // Verifica se é 1D (por enquanto)
        if shape.len() != 1 {
            return Err(anyhow!("Apenas kernels 1D são suportados atualmente (shape.len() = {})", shape.len()));
        }
        // Confere se n é igual ao produto das dimensões
        let expected_n = shape.iter().product::<usize>() as i32;
        if n != expected_n {
            return Err(anyhow!(
                "n ({}) não corresponde ao produto das dimensões de shape ({:?}) = {}",
                n, shape, expected_n
            ));
        }

        // --- 2. Coletar identidade da GPU (com Result) ---
        let gpu = GpuIdentity::from_system()
            .map_err(|e| anyhow!("Falha ao detectar GPU: {}", e))?;
        eprintln!("[Interceptor] GPU: {:?}", gpu);

        // --- 3. Metadados para hash ---
        // ATENÇÃO: NÃO inclui job_id/node_id na chave de cache
        let mut meta = KernelMetadata {
            operation: op.clone(),
            dtype: dtype.clone(),
            shape: shape.clone(),
            vendor: gpu.vendor.clone(),
            arch: gpu.arch.clone(),
            driver_version: gpu.driver_version.clone(),
            job_id: None,      // NÃO usado na cache
            node_id: None,     // NÃO usado na cache
            capabilities: gpu.capabilities.clone(),
        };

        // --- 4. Chave de cache (BLAKE3) ---
        let cache_key = meta.cache_key();
        eprintln!("[Interceptor] Chave cache (BLAKE3): {}", cache_key);

        // --- 5. Tentar cache L1 ---
        if let Some(cached) = self.local_cache.get(&cache_key) {
            eprintln!("[Interceptor] Cache L1 HIT");
            // Recuperamos os parâmetros de lançamento salvos junto com o binário
            let tile_size = cached.tile_size;
            let shared_mem_bytes = cached.shared_mem_bytes;
            let radius = cached.radius;
            // Executa imediatamente
            return self.execute_kernel(
                &cached.binary,
                device_ptr_x,
                device_ptr_y,
                n,
                tile_size,
                shared_mem_bytes,
                radius,
                Some(cache_key.clone()),
            );
        } else {
            eprintln!("[Interceptor] Cache L1 MISS");
        }

        // --- 6. (Opcional) Cache L2 (Redis) – ainda não implementado ---
        // if let Some(redis_bin) = cluster_cache::get(&cache_key) { ... }

        // --- 7. Compilação do zero ---
        eprintln!("[Interceptor] Compilando do zero...");

        // 7.1 Construir FLIR (recebe dtype)
        let flir_module = build_flir("", &gpu.capabilities, &dtype)
            .map_err(|e| anyhow!("Falha ao construir FLIR: {}", e))?;

        let flir_op = flir_module.ops.first()
            .ok_or_else(|| anyhow!("Módulo FLIR sem operações"))?;
        let flir_params = flir_op.params.as_ref()
            .ok_or_else(|| anyhow!("Operação sem params"))?;

        // Extrai parâmetros que serão salvos com o binário
        let radius = flir_params["radius"].as_i64().unwrap_or(1) as u32;
        let tile_size = flir_params["tile_size"].as_i64().unwrap_or(128) as u32;
        let shared_mem_bytes = flir_params["shared_mem_bytes"].as_u64().unwrap_or(0) as u32;

        // 7.2 Gerar LLVM IR (passando dtype)
        let llvm_ir = flir_to_llvm(&flir_module, &gpu.capabilities, &dtype)
            .map_err(|e| anyhow!("Falha ao gerar LLVM IR: {}", e))?;

        // 7.3 Compilar para PTX
        let ptx_bytes = compile_to_ptx(&llvm_ir, &gpu.capabilities)
            .map_err(|e| anyhow!("Falha ao compilar para PTX: {}", e))?;

        // 7.4 Salvar no cache L1 (com parâmetros)
        let cached_entry = local_cache::CachedKernel {
            binary: ptx_bytes.clone(),
            target: "ptx".to_string(),
            tile_size,
            shared_mem_bytes,
            radius,
        };
        self.local_cache.set(&cache_key, &cached_entry);

        // --- 8. Executar o kernel ---
        self.execute_kernel(
            &ptx_bytes,
            device_ptr_x,
            device_ptr_y,
            n,
            tile_size,
            shared_mem_bytes,
            radius,
            Some(cache_key.clone()),
        )
    }

    // ----------------------------------------------------------------------
    // Função auxiliar que executa o kernel
    // ----------------------------------------------------------------------
    fn execute_kernel(
        &self,
        ptx_bytes: &[u8],
        device_ptr_x: *mut c_void,
        device_ptr_y: *mut c_void,
        n: i32,
        tile_size: u32,
        shared_mem_bytes: u32,
        radius: u32,
        kernel_hash: Option<String>,
    ) -> Result<()> {
        // Monta parâmetros FLIR (usando os valores recebidos)
        let flir_params = serde_json::json!({
            "tile_size": tile_size,
            "shared_mem_bytes": shared_mem_bytes,
            "radius": radius,
        });

        let input_ptrs = vec![device_ptr_x as *const c_void];
        let output_ptrs = vec![device_ptr_y as *const c_void];

        // Executa via executor
        execute_flir_kernel(
            ptx_bytes,
            "basalto_kernel",
            &flir_params,
            &input_ptrs,
            &output_ptrs,
            n,
            kernel_hash.clone(),
            self.jit_sender.clone(),
        )?;

        // Se auditoria habilitada, gerar SHA-256 (usando job_id/node_id separadamente)
        if std::env::var("BASALTO_AUDIT_ENABLED").unwrap_or_default() == "true" {
            // Precisamos do job_id/node_id – eles não estão em meta (evitamos contaminação)
            // Vamos buscá-los novamente da GPU (ou de variáveis de ambiente)
            let gpu = GpuIdentity::from_system()?;
            let job_id = std::env::var("SLURM_JOB_ID").ok();
            let audit_meta = KernelMetadata {
                operation: "stencil_1d".to_string(),
                dtype: "f64".to_string(), // seria o dtype real, mas não temos aqui
                shape: vec![n as usize],
                vendor: gpu.vendor.clone(),
                arch: gpu.arch.clone(),
                driver_version: gpu.driver_version.clone(),
                job_id,
                node_id: Some(gpu.node_id.clone()),
                capabilities: gpu.capabilities.clone(),
            };
            let audit_digest = audit_meta.audit_digest();
            eprintln!("[Interceptor] Audit SHA-256: {}", audit_digest);
            // TODO: enviar para energy-telemetry
        }

        Ok(())
    }
}

// --------------------------------------------------------------------------
// Bindings PyO3
// --------------------------------------------------------------------------
use pyo3::prelude::*;

#[pyclass]
pub struct PyBasaltoInterceptor {
    inner: BasaltoInterceptor,
}

#[pymethods]
impl PyBasaltoInterceptor {
    #[new]
    pub fn new() -> Self {
        let sender: Option<mpsc::Sender<KernelExecutionReport>> = None;
        let inner = BasaltoInterceptor::new(sender);
        Self { inner }
    }

    pub fn compile_and_execute(
        &self,
        py: Python<'_>,
        op: String,
        dtype: String,
        shape: Vec<usize>,
        job_id: Option<String>,
        device_ptr_x: usize,
        device_ptr_y: usize,
        n: i32,
    ) -> PyResult<()> {
        let ptr_x = device_ptr_x as *mut c_void;
        let ptr_y = device_ptr_y as *mut c_void;

        // Libera o GIL durante o trabalho pesado
        py.allow_threads(|| {
            self.inner.compile_and_execute(
                op,
                dtype,
                shape,
                job_id,
                ptr_x,
                ptr_y,
                n,
            )
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

#[pymodule]
pub fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBasaltoInterceptor>()?;
    Ok(())
}