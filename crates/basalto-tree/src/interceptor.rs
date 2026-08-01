// crates/basalto-tree/src/interceptor.rs
use anyhow::{anyhow, Result};
use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use dashmap::DashMap;

use basalto_common::hardware::GpuIdentity;
use basalto_core::hasher::KernelMetadata;
use basalto_core::flir_builder::{build_flir, flir_to_llvm, compile_to_ptx};
use basalto_tree::local_cache::{self, LocalCache};
use basalto_tree::executor::{execute_flir_kernel, KernelExecutionReport};

// --------------------------------------------------------------------------
// Guard RAII para remover a entrada do mapa de compilações em andamento
// --------------------------------------------------------------------------
struct InFlightGuard {
    key: String,
    map: Arc<DashMap<String, Arc<Mutex<()>>>>,
}

impl InFlightGuard {
    fn new(map: &Arc<DashMap<String, Arc<Mutex<()>>>>, key: String) -> Self {
        Self { key, map: Arc::clone(map) }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.map.remove(&self.key);
    }
}

// --------------------------------------------------------------------------
// Interceptor principal
// --------------------------------------------------------------------------
pub struct BasaltoInterceptor {
    local_cache: LocalCache,
    jit_sender: Option<mpsc::Sender<KernelExecutionReport>>,
    in_flight: Arc<DashMap<String, Arc<Mutex<()>>>>,
}

impl BasaltoInterceptor {
    pub fn new(jit_sender: Option<mpsc::Sender<KernelExecutionReport>>) -> Self {
        Self {
            local_cache: LocalCache::new_with_capacity(10_000),
            jit_sender,
            in_flight: Arc::new(DashMap::new()),
        }
    }

    /// Ponto de entrada principal — chamado pelo executor do Rust.
    pub fn compile_and_execute(
        &self,
        op: String,
        dtype: String,
        shape: Vec<usize>,
        job_id: Option<String>,
        device_ptr_x: *mut c_void,
        device_ptr_y: *mut c_void,
    ) -> Result<()> {
        // ------------------------------------------------------------------
        // 1. Validações de entrada (1D, 2D ou 3D)
        // ------------------------------------------------------------------
        if shape.is_empty() || shape.len() > 3 {
            return Err(anyhow!("Apenas 1D, 2D e 3D são suportados (shape = {:?})", shape));
        }
        if device_ptr_x.is_null() || device_ptr_y.is_null() {
            return Err(anyhow!("Ponteiros de dispositivo nulos"));
        }

        // ------------------------------------------------------------------
        // 2. Coletar identidade da GPU
        // ------------------------------------------------------------------
        let gpu = GpuIdentity::from_system()
            .map_err(|e| anyhow!("Falha ao detectar GPU: {}", e))?;
        eprintln!("[Interceptor] GPU: vendor={}, arch={}, driver={}",
            gpu.vendor, gpu.arch, gpu.driver_version);

        // ------------------------------------------------------------------
        // 3. Metadados para chave de cache (NÃO inclui job_id/node_id)
        // ------------------------------------------------------------------
        let meta = KernelMetadata {
            operation: op.clone(),
            dtype: dtype.clone(),
            shape: shape.clone(),
            vendor: gpu.vendor.clone(),
            arch: gpu.arch.clone(),
            driver_version: gpu.driver_version.clone(),
            job_id: None,
            node_id: None,
            capabilities: gpu.capabilities.clone(),
        };

        let cache_key = meta.cache_key();
        eprintln!("[Interceptor] Chave cache (BLAKE3): {}", cache_key);

        // ------------------------------------------------------------------
        // 4. Tentar cache L1 (LRU)
        // ------------------------------------------------------------------
        if let Some(cached) = self.local_cache.get(&cache_key) {
            eprintln!("[Interceptor] Cache L1 HIT");
            let tile_x = cached.tile_x.unwrap_or(128);
            let tile_y = cached.tile_y.unwrap_or(1);
            let shared_mem_bytes = cached.shared_mem_bytes;
            let flir_params = serde_json::json!({
                "tile_x": tile_x,
                "tile_y": tile_y,
                "shared_mem_bytes": shared_mem_bytes,
            });
            return execute_flir_kernel(
                &cached.binary,
                "basalto_kernel",
                &flir_params,
                &[device_ptr_x as *const c_void],
                &[device_ptr_y as *const c_void],
                &shape,
                Some(cache_key),
                self.jit_sender.clone(),
            );
        }
        eprintln!("[Interceptor] Cache L1 MISS");

        // ------------------------------------------------------------------
        // 5. (Opcional) Cache L2 (Redis) – ainda não implementado
        // ------------------------------------------------------------------

        // ------------------------------------------------------------------
        // 6. Controle de compilações concorrentes (anti-thundering-herd)
        // ------------------------------------------------------------------
        let lock: Arc<Mutex<()>> = self.in_flight
            .entry(cache_key.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();

        let _in_flight_guard = InFlightGuard::new(&self.in_flight, cache_key.clone());
        let _lock_guard = lock.lock().unwrap();

        if let Some(cached) = self.local_cache.get(&cache_key) {
            eprintln!("[Interceptor] Cache preenchido por outra thread durante a espera.");
            let flir_params = serde_json::json!({
                "tile_x": cached.tile_x.unwrap_or(128),
                "tile_y": cached.tile_y.unwrap_or(1),
                "shared_mem_bytes": cached.shared_mem_bytes,
            });
            return execute_flir_kernel(
                &cached.binary,
                "basalto_kernel",
                &flir_params,
                &[device_ptr_x as *const c_void],
                &[device_ptr_y as *const c_void],
                &shape,
                Some(cache_key),
                self.jit_sender.clone(),
            );
        }

        // ------------------------------------------------------------------
        // 7. Compilação do zero
        // ------------------------------------------------------------------
        eprintln!("[Interceptor] Compilando do zero...");

        let flir_module = build_flir("", &gpu.capabilities, &dtype, &shape)
            .map_err(|e| anyhow!("Falha ao construir FLIR: {}", e))?;

        let flir_op = flir_module.ops.first()
            .ok_or_else(|| anyhow!("Módulo FLIR sem operações"))?;
        let flir_params = flir_op.params.as_ref()
            .ok_or_else(|| anyhow!("Operação sem params"))?;

        let tile_x = flir_params["tile_x"].as_i64().unwrap_or(128) as u32;
        let tile_y = flir_params["tile_y"].as_i64().unwrap_or(1) as u32;
        let shared_mem_bytes = flir_params["shared_mem_bytes"].as_u64().unwrap_or(0) as u32;

        let llvm_ir = flir_to_llvm(&flir_module, &gpu.capabilities, &dtype)
            .map_err(|e| anyhow!("Falha ao gerar LLVM IR: {}", e))?;

        let ptx_bytes = compile_to_ptx(&llvm_ir, &gpu.capabilities)
            .map_err(|e| anyhow!("Falha ao compilar para PTX: {}", e))?;

        let cached_entry = local_cache::CachedKernel {
            binary: ptx_bytes.clone(),
            target: "ptx".to_string(),
            tile_x: Some(tile_x),
            tile_y: Some(tile_y),
            shared_mem_bytes,
            radius: 1,
        };
        self.local_cache.set(&cache_key, &cached_entry);

        // ------------------------------------------------------------------
        // 8. Executar o kernel
        // ------------------------------------------------------------------
        let flir_params_for_exec = serde_json::json!({
            "tile_x": tile_x,
            "tile_y": tile_y,
            "shared_mem_bytes": shared_mem_bytes,
        });
        execute_flir_kernel(
            &ptx_bytes,
            "basalto_kernel",
            &flir_params_for_exec,
            &[device_ptr_x as *const c_void],
            &[device_ptr_y as *const c_void],
            &shape,
            Some(cache_key),
            self.jit_sender.clone(),
        )?;

        // ------------------------------------------------------------------
        // 9. Auditoria (SHA‑256)
        // ------------------------------------------------------------------
        if std::env::var("BASALTO_AUDIT_ENABLED").unwrap_or_default() == "true" {
            let effective_job_id = job_id.or_else(|| {
                std::env::var("SLURM_JOB_ID")
                    .or_else(|_| std::env::var("PBS_JOBID"))
                    .or_else(|_| std::env::var("LSB_JOBID"))
                    .ok()
            });

            let audit_meta = KernelMetadata {
                operation: op,
                dtype,
                shape: shape.clone(),
                vendor: gpu.vendor.clone(),
                arch: gpu.arch.clone(),
                driver_version: gpu.driver_version.clone(),
                job_id: effective_job_id,
                node_id: Some(gpu.node_id.clone()),
                capabilities: gpu.capabilities.clone(),
            };
            let audit_digest = audit_meta.audit_digest();
            eprintln!("[Interceptor] Audit SHA-256: {}", audit_digest);
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
    ) -> PyResult<()> {
        let ptr_x = device_ptr_x as *mut c_void;
        let ptr_y = device_ptr_y as *mut c_void;

        py.allow_threads(|| {
            self.inner.compile_and_execute(
                op,
                dtype,
                shape,
                job_id,
                ptr_x,
                ptr_y,
            )
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

#[pymodule]
pub fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBasaltoInterceptor>()?;
    Ok(())
}