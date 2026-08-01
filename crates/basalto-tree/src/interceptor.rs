use anyhow::{anyhow, Result};
use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use dashmap::DashMap;

use basalto_common::hardware::{GpuIdentity, DeviceCapabilities};
use basalto_core::hasher::KernelMetadata;
use basalto_core::flir_builder::{build_flir, flir_to_llvm, compile_to_ptx};
use basalto_tree::local_cache::{self, LocalCache};
use basalto_tree::executor::{execute_flir_kernel, KernelExecutionReport};
use energy_telemetry::correlator::Correlator;
use energy_telemetry::comparator::TemporalComparator;
use siliconforge_jit::{SiliconForgeProfiler, SiliconForgeOptimizer, SiliconForgeCompiler};
use siliconforge_jit::profiler::KernelExecutionRecord;
use basalto_communication::{MpiRuntime, NcclRuntime, CudaRuntime, HaloExchanger};

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

pub struct BasaltoInterceptor {
    local_cache: Arc<LocalCache>,
    jit_sender: Option<mpsc::Sender<KernelExecutionReport>>,
    in_flight: Arc<DashMap<String, Arc<Mutex<()>>>>,
    correlator: Arc<Correlator>,
    comparator: Arc<TemporalComparator>,
    profiler_sender: Option<mpsc::Sender<KernelExecutionRecord>>,
    halo_exchanger: Option<Arc<HaloExchanger>>,
    _siliconforge_handle: Option<tokio::task::JoinHandle<()>>,
}

impl BasaltoInterceptor {
    pub fn new(jit_sender: Option<mpsc::Sender<KernelExecutionReport>>) -> Self {
        let correlator = Arc::new(Correlator::new());
        let comparator = Arc::new(TemporalComparator::new(correlator.clone()));
        let local_cache = Arc::new(LocalCache::new_with_capacity(10_000));

        // ================================================================
        // INICIALIZAÇÃO DO MPI / NCCL / CUDA PARA TROCA DE HALOS
        // ================================================================
        let mpi = match MpiRuntime::new() {
            Ok(m) => Arc::new(m),
            Err(e) => {
                eprintln!("[Interceptor] MPI não disponível: {}", e);
                // Retorna um interceptor sem suporte a comunicação (modo single-node)
                let (profiler_tx, _) = mpsc::channel(1);
                let profiler = Arc::new(SiliconForgeProfiler::new());
                let gpu = GpuIdentity::from_system().unwrap_or_default();
                let caps = gpu.capabilities.clone().unwrap_or(DeviceCapabilities {
                    compute_capability_major: 7,
                    compute_capability_minor: 0,
                    max_threads_per_block: 1024,
                    max_shared_memory_per_block: 49152,
                    max_registers_per_block: 65536,
                    warp_size: 32,
                    multi_processor_count: 80,
                });
                let optimizer = Arc::new(SiliconForgeOptimizer::new(caps));
                let gpu_identity = Arc::new(gpu);
                let compiler = Arc::new(SiliconForgeCompiler::new(
                    profiler.clone(),
                    optimizer.clone(),
                    local_cache.clone(),
                    gpu_identity.clone(),
                ));
                let handle = tokio::spawn(async move {
                    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
                    loop {
                        tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
                        let profiles = profiler.get_all_profiles();
                        for profile in &profiles {
                            let suggestions = optimizer.analyze(profile);
                            for suggestion in suggestions {
                                if suggestion.confidence > 0.6 {
                                    if let Err(e) = compiler.process_suggestion(suggestion).await {
                                        eprintln!("[SiliconForge] Erro ao aplicar otimização: {}", e);
                                    }
                                }
                            }
                        }
                    }
                });
                return Self {
                    local_cache,
                    jit_sender,
                    in_flight: Arc::new(DashMap::new()),
                    correlator,
                    comparator,
                    profiler_sender: Some(profiler_tx),
                    halo_exchanger: None,
                    _siliconforge_handle: Some(handle),
                };
            }
        };

        let nccl = match NcclRuntime::new() {
            Ok(n) => Some(Arc::new(n)),
            Err(e) => {
                eprintln!("[Interceptor] NCCL não disponível: {}", e);
                None
            }
        };

        let cuda = match CudaRuntime::new() {
            Ok(c) => Arc::new(c),
            Err(e) => {
                eprintln!("[Interceptor] CUDA Runtime não disponível: {}", e);
                panic!("CUDA Runtime é obrigatória para troca de halos");
            }
        };

        let halo_exchanger = match HaloExchanger::new(mpi, nccl, cuda) {
            Ok(h) => Some(Arc::new(h)),
            Err(e) => {
                eprintln!("[Interceptor] Erro ao criar HaloExchanger: {}", e);
                None
            }
        };

        // ================================================================
        // INICIALIZAÇÃO DO SILICONFORGE JIT
        // ================================================================
        let (profiler_tx, mut profiler_rx) = mpsc::channel::<KernelExecutionRecord>(10000);
        let profiler = Arc::new(SiliconForgeProfiler::new());

        let gpu = GpuIdentity::from_system().unwrap_or_default();
        let caps = gpu.capabilities.clone().unwrap_or(DeviceCapabilities {
            compute_capability_major: 7,
            compute_capability_minor: 0,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
            max_registers_per_block: 65536,
            warp_size: 32,
            multi_processor_count: 80,
        });

        let optimizer = Arc::new(SiliconForgeOptimizer::new(caps));
        let gpu_identity = Arc::new(gpu);
        let compiler = Arc::new(SiliconForgeCompiler::new(
            profiler.clone(),
            optimizer.clone(),
            local_cache.clone(),
            gpu_identity.clone(),
        ));

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
            loop {
                tokio::select! {
                    Some(record) = profiler_rx.recv() => {
                        profiler.record(record);
                    }
                    _ = interval.tick() => {
                        let profiles = profiler.get_all_profiles();
                        for profile in &profiles {
                            let suggestions = optimizer.analyze(profile);
                            for suggestion in suggestions {
                                if suggestion.confidence > 0.6 {
                                    if let Err(e) = compiler.process_suggestion(suggestion).await {
                                        eprintln!("[SiliconForge] Erro ao aplicar otimização: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        Self {
            local_cache,
            jit_sender,
            in_flight: Arc::new(DashMap::new()),
            correlator,
            comparator,
            profiler_sender: Some(profiler_tx),
            halo_exchanger,
            _siliconforge_handle: Some(handle),
        }
    }

    /// Extrai o `radius` da operação. Exemplo: "stencil_3d_r4" -> radius=4
    fn extract_radius(op: &str) -> usize {
        if let Some(r) = op.split("_r").nth(1) {
            if let Ok(radius) = r.parse::<usize>() {
                return radius;
            }
        }
        if op.contains("order_8") { return 4; }
        if op.contains("order_12") { return 6; }
        1
    }

    pub fn compile_and_execute(
        &self,
        op: String,
        dtype: String,
        shape: Vec<usize>,
        strides: Vec<isize>,
        job_id: Option<String>,
        device_ptr_x: *mut c_void,
        device_ptr_y: *mut c_void,
    ) -> Result<()> {
        if shape.is_empty() || shape.len() > 3 {
            return Err(anyhow!("Apenas 1D, 2D e 3D são suportados (shape = {:?})", shape));
        }
        if shape.len() != strides.len() {
            return Err(anyhow!("Shape e strides devem ter o mesmo comprimento"));
        }
        if device_ptr_x.is_null() || device_ptr_y.is_null() {
            return Err(anyhow!("Ponteiros de dispositivo nulos"));
        }

        let radius = Self::extract_radius(&op);
        eprintln!("[Interceptor] Radius detectado: {} (ordem {})", radius, 2 * radius);

        let gpu = GpuIdentity::from_system()
            .map_err(|e| anyhow!("Falha ao detectar GPU: {}", e))?;
        eprintln!("[Interceptor] GPU: vendor={}, arch={}, driver={}",
            gpu.vendor, gpu.arch, gpu.driver_version);

        let meta = KernelMetadata {
            operation: op.clone(),
            dtype: dtype.clone(),
            shape: shape.clone(),
            strides: strides.clone(),
            radius,
            vendor: gpu.vendor.clone(),
            arch: gpu.arch.clone(),
            driver_version: gpu.driver_version.clone(),
            job_id: None,
            node_id: None,
            capabilities: gpu.capabilities.clone(),
        };

        let cache_key = meta.cache_key();
        eprintln!("[Interceptor] Chave cache (BLAKE3): {}", cache_key);

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
                &strides,
                radius,
                Some(cache_key),
                self.jit_sender.clone(),
                self.profiler_sender.clone(),
                self.correlator.clone(),
                self.comparator.clone(),
                &op,
                &dtype,
                job_id.as_deref(),
                self.halo_exchanger.clone(),
                device_ptr_x,
                device_ptr_y,
            );
        }
        eprintln!("[Interceptor] Cache L1 MISS");

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
                &strides,
                radius,
                Some(cache_key),
                self.jit_sender.clone(),
                self.profiler_sender.clone(),
                self.correlator.clone(),
                self.comparator.clone(),
                &op,
                &dtype,
                job_id.as_deref(),
                self.halo_exchanger.clone(),
                device_ptr_x,
                device_ptr_y,
            );
        }

        eprintln!("[Interceptor] Compilando do zero...");

        let flir_module = build_flir("", &gpu.capabilities, &dtype, &shape, &strides, radius)
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
            radius: radius as u32,
            metadata: Some(meta.clone()),
        };
        self.local_cache.set(&cache_key, &cached_entry);

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
            &strides,
            radius,
            Some(cache_key),
            self.jit_sender.clone(),
            self.profiler_sender.clone(),
            self.correlator.clone(),
            self.comparator.clone(),
            &op,
            &dtype,
            job_id.as_deref(),
            self.halo_exchanger.clone(),
            device_ptr_x,
            device_ptr_y,
        )?;

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
                strides: strides.clone(),
                radius,
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
        strides: Vec<isize>,
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
                strides,
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