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
// ao final do escopo (sucesso, erro ou early return).
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
        n: i32,
    ) -> Result<()> {
        // ------------------------------------------------------------------
        // 1. Validações de entrada (shape 1D, n consistente)
        // ------------------------------------------------------------------
        if shape.len() != 1 {
            return Err(anyhow!(
                "Apenas kernels 1D são suportados atualmente (shape.len() = {})",
                shape.len()
            ));
        }
        let expected_n = shape.iter().product::<usize>() as i32;
        if n != expected_n {
            return Err(anyhow!(
                "n ({}) não corresponde ao produto das dimensões de shape ({:?}) = {}",
                n, shape, expected_n
            ));
        }

        // ------------------------------------------------------------------
        // 2. Coletar identidade da GPU (UMA ÚNICA VEZ)
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
            return self.execute_kernel(
                &cached.binary,
                &gpu,
                &op,
                &dtype,
                job_id,
                device_ptr_x,
                device_ptr_y,
                n,
                cached.tile_size,
                cached.shared_mem_bytes,
                cached.radius,
                Some(cache_key),
            );
        }
        eprintln!("[Interceptor] Cache L1 MISS");

        // ------------------------------------------------------------------
        // 5. (Opcional) Cache L2 (Redis) – ainda não implementado
        // ------------------------------------------------------------------

        // ------------------------------------------------------------------
        // 6. Controle de compilações concorrentes (anti-thundering-herd)
        // ------------------------------------------------------------------
        // Obtém (ou cria) o Arc<Mutex<()>> para esta chave. O `.clone()` imediato
        // libera o guard do DashMap, evitando travar outros shards durante a compilação.
        let lock: Arc<Mutex<()>> = self.in_flight
            .entry(cache_key.clone())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone();

        // Cria o guard RAII que removerá a entrada no final do escopo,
        // independente do caminho de saída (sucesso, erro, early return).
        let _in_flight_guard = InFlightGuard::new(&self.in_flight, cache_key.clone());

        // Adquire o lock – se outra thread já estiver compilando a mesma chave, aguarda.
        let _lock_guard = lock.lock().unwrap();

        // Double-check: após obter o lock, verifica se o cache foi preenchido
        // por outra thread durante a espera.
        if let Some(cached) = self.local_cache.get(&cache_key) {
            eprintln!("[Interceptor] Cache preenchido por outra thread durante a espera.");
            return self.execute_kernel(
                &cached.binary,
                &gpu,
                &op,
                &dtype,
                job_id,
                device_ptr_x,
                device_ptr_y,
                n,
                cached.tile_size,
                cached.shared_mem_bytes,
                cached.radius,
                Some(cache_key),
            );
        }

        // ------------------------------------------------------------------
        // 7. Compilação do zero (apenas uma thread entra aqui)
        // ------------------------------------------------------------------
        eprintln!("[Interceptor] Compilando do zero...");

        // 7.1 Construir FLIR (passando dtype)
        let flir_module = build_flir("", &gpu.capabilities, &dtype)
            .map_err(|e| anyhow!("Falha ao construir FLIR: {}", e))?;

        // 7.2 Extrair parâmetros de lançamento (tile_size, shared_mem, radius)
        let flir_op = flir_module.ops.first()
            .ok_or_else(|| anyhow!("Módulo FLIR sem operações"))?;
        let flir_params = flir_op.params.as_ref()
            .ok_or_else(|| anyhow!("Operação sem params"))?;

        let radius = flir_params["radius"].as_i64().unwrap_or(1) as u32;
        let tile_size = flir_params["tile_size"].as_i64().unwrap_or(128) as u32;
        let shared_mem_bytes = flir_params["shared_mem_bytes"].as_u64().unwrap_or(0) as u32;

        // 7.3 Gerar LLVM IR (passando dtype)
        let llvm_ir = flir_to_llvm(&flir_module, &gpu.capabilities, &dtype)
            .map_err(|e| anyhow!("Falha ao gerar LLVM IR: {}", e))?;

        // 7.4 Compilar para PTX
        let ptx_bytes = compile_to_ptx(&llvm_ir, &gpu.capabilities)
            .map_err(|e| anyhow!("Falha ao compilar para PTX: {}", e))?;

        // 7.5 Salvar no cache L1 (com todos os parâmetros de lançamento)
        let cached_entry = local_cache::CachedKernel {
            binary: ptx_bytes.clone(),
            target: "ptx".to_string(),
            tile_size,
            shared_mem_bytes,
            radius,
        };
        self.local_cache.set(&cache_key, &cached_entry);

        // ------------------------------------------------------------------
        // 8. Executar o kernel
        // ------------------------------------------------------------------
        self.execute_kernel(
            &ptx_bytes,
            &gpu,
            &op,
            &dtype,
            job_id,
            device_ptr_x,
            device_ptr_y,
            n,
            tile_size,
            shared_mem_bytes,
            radius,
            Some(cache_key),
        )
    }

    // ----------------------------------------------------------------------
    // Função auxiliar de execução – recebe todos os parâmetros necessários
    // ----------------------------------------------------------------------
    #[allow(clippy::too_many_arguments)]
    fn execute_kernel(
        &self,
        ptx_bytes: &[u8],
        gpu: &GpuIdentity,
        op: &str,
        dtype: &str,
        job_id: Option<String>,
        device_ptr_x: *mut c_void,
        device_ptr_y: *mut c_void,
        n: i32,
        tile_size: u32,
        shared_mem_bytes: u32,
        radius: u32,
        kernel_hash: Option<String>,
    ) -> Result<()> {
        // Monta parâmetros para o executor (FLIR)
        let flir_params = serde_json::json!({
            "tile_size": tile_size,
            "shared_mem_bytes": shared_mem_bytes,
            "radius": radius,
        });

        let input_ptrs = vec![device_ptr_x as *const c_void];
        let output_ptrs = vec![device_ptr_y as *const c_void];

        // Dispara o kernel via executor (usa o runtime CUDA)
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

        // ------------------------------------------------------------------
        // Auditoria (SHA‑256) – se habilitada
        // ------------------------------------------------------------------
        if std::env::var("BASALTO_AUDIT_ENABLED").unwrap_or_default() == "true" {
            // Usa o job_id recebido, ou fallback para variáveis de ambiente
            // (suporte a SLURM, PBS, LSF de forma agnóstica).
            let effective_job_id = job_id.or_else(|| {
                std::env::var("SLURM_JOB_ID")
                    .or_else(|_| std::env::var("PBS_JOBID"))
                    .or_else(|_| std::env::var("LSB_JOBID"))
                    .ok()
            });

            let audit_meta = KernelMetadata {
                operation: op.to_string(),
                dtype: dtype.to_string(),
                shape: vec![n as usize],
                vendor: gpu.vendor.clone(),
                arch: gpu.arch.clone(),
                driver_version: gpu.driver_version.clone(),
                job_id: effective_job_id,
                node_id: Some(gpu.node_id.clone()),
                capabilities: gpu.capabilities.clone(),
            };

            let audit_digest = audit_meta.audit_digest();
            eprintln!("[Interceptor] Audit SHA-256: {}", audit_digest);
            // TODO: enviar para energy-telemetry com (audit_digest, timestamp, kWh)
        }

        Ok(())
    }
}

// --------------------------------------------------------------------------
// Bindings PyO3 – expõe o interceptor ao Python
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
        // Em produção, o canal para o JIT seria injetado aqui.
        let sender: Option<mpsc::Sender<KernelExecutionReport>> = None;
        let inner = BasaltoInterceptor::new(sender);
        Self { inner }
    }

    /// Método chamado pelo Python (via PyTorch)
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

        // Libera o GIL durante o trabalho pesado (compilação + execução)
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
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

/// Registra o módulo Python `_rust` (chamado a partir do `compiler.py`)
#[pymodule]
pub fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBasaltoInterceptor>()?;
    Ok(())
}