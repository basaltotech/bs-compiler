// crates/basalto-tree/src/interceptor.rs
use anyhow::{anyhow, Result};
use std::ffi::c_void;
use std::sync::Arc;
use tokio::sync::mpsc;
use serde_json::Value;

// Crates internos
use basalto_common::hardware::GpuIdentity;
use basalto_core::hasher::KernelMetadata;
use basalto_core::flir_builder::{build_flir, flir_to_llvm, compile_to_ptx};
use basalto_tree::local_cache::LocalCache;
use basalto_tree::executor::{execute_flir_kernel, KernelExecutionReport};
use basalto_tree::cluster_cache::ClusterCache; // será implementado depois

// --------------------------------------------------------------------------
// Interceptor principal – orquestra todo o fluxo
// --------------------------------------------------------------------------
pub struct BasaltoInterceptor {
    local_cache: LocalCache,
    // cluster_cache: ClusterCache, // opcional
    // sender para o SiliconForge JIT
    jit_sender: Option<mpsc::Sender<KernelExecutionReport>>,
}

impl BasaltoInterceptor {
    /// Cria um novo interceptor.
    /// `jit_sender` é opcional – se fornecido, o executor enviará relatórios de execução
    /// para o SiliconForge JIT rodar em background.
    pub fn new(jit_sender: Option<mpsc::Sender<KernelExecutionReport>>) -> Self {
        Self {
            local_cache: LocalCache::new(),
            jit_sender,
        }
    }

    /// Ponto de entrada principal – chamado pelo Python via PyO3.
    ///
    /// # Parâmetros (recebidos do PyTorch via PyO3)
    /// - `op`: string com o nome da operação ("stencil_1d", "matmul", etc.)
    /// - `dtype`: string ("f32", "f64", "bf16")
    /// - `shape`: vetor de dimensões (ex: [1024])
    /// - `job_id`: opcional – SLURM_JOB_ID ou similar
    /// - `device_ptr_x`: ponteiro para o tensor de entrada na GPU (CUDA device pointer)
    /// - `device_ptr_y`: ponteiro para o tensor de saída na GPU
    /// - `n`: número de elementos (ou tamanho do array)
    ///
    /// # Retorno
    /// - `Ok(())` se o kernel compilou e executou com sucesso.
    /// - `Err` com mensagem descritiva se algo falhar.
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
        eprintln!("[Interceptor] Iniciando compilação/execução para op={}, dtype={}, shape={:?}", op, dtype, shape);

        // ------------------------------------------------------------------
        // 1. Coletar identidade da GPU (vendor, arch, driver, capabilities)
        //    Usa root para ler DeviceCapabilities via CUDA API.
        // ------------------------------------------------------------------
        let gpu = GpuIdentity::from_system();
        eprintln!("[Interceptor] GPU: {:?}", gpu);

        // ------------------------------------------------------------------
        // 2. Construir metadados para hash
        // ------------------------------------------------------------------
        let mut meta = KernelMetadata {
            operation: op.clone(),
            dtype: dtype.clone(),
            shape: shape.clone(),
            vendor: gpu.vendor.clone(),
            arch: gpu.arch.clone(),
            driver_version: gpu.driver_version.clone(),
            job_id: job_id.clone(),
            node_id: Some(gpu.node_id.clone()),
            capabilities: gpu.capabilities.clone(),
        };

        // ------------------------------------------------------------------
        // 3. Calcular chave de cache (BLAKE3)
        // ------------------------------------------------------------------
        let cache_key = meta.cache_key();
        eprintln!("[Interceptor] Chave de cache (BLAKE3): {}", cache_key);

        // ------------------------------------------------------------------
        // 4. Tentar cache L1 (disco local)
        // ------------------------------------------------------------------
        if let Some(cached) = self.local_cache.get(&cache_key) {
            eprintln!("[Interceptor] Cache L1 HIT! Carregando binário pré-compilado.");
            let ptx_bytes = cached.binary;
            // Executa imediatamente com o binário cacheado
            return self.execute_kernel(
                &ptx_bytes,
                &meta,
                device_ptr_x,
                device_ptr_y,
                n,
                Some(cache_key.clone()),
            );
        } else {
            eprintln!("[Interceptor] Cache L1 MISS.");
        }

        // ------------------------------------------------------------------
        // 5. Opcional: Tentar cache L2 (Redis) – oportunista, não bloqueante
        //    (será implementado depois em cluster_cache.rs)
        // ------------------------------------------------------------------
        // if let Some(redis_bin) = ClusterCache::get(&cache_key) {
        //     eprintln!("[Interceptor] Cache L2 (Redis) HIT!");
        //     self.local_cache.set(&cache_key, redis_bin);
        //     return self.execute_kernel(&redis_bin, &meta, device_ptr_x, device_ptr_y, n, Some(cache_key));
        // }

        // ------------------------------------------------------------------
        // 6. Cache miss – compilar do zero
        // ------------------------------------------------------------------
        eprintln!("[Interceptor] Cache miss – iniciando compilação...");

        // 6.1 Construir FLIR a partir do grafo (placeholder – ainda recebe string vazia)
        //     No futuro, o Python passará a representação serializada do grafo FX.
        let graph_str = ""; // TODO: receber do Python
        let flir_module = build_flir(graph_str, &gpu.capabilities)
            .map_err(|e| anyhow!("Falha ao construir FLIR: {}", e))?;

        // 6.2 Extrair parâmetros da operação (tile_size, shared_mem_bytes, etc.)
        let flir_op = flir_module.ops.first()
            .ok_or_else(|| anyhow!("Módulo FLIR sem operações"))?;
        let flir_params = flir_op.params.as_ref()
            .ok_or_else(|| anyhow!("Operação FLIR sem parâmetros"))?;

        // 6.3 Gerar LLVM IR
        let llvm_ir = flir_to_llvm(&flir_module, &gpu.capabilities)
            .map_err(|e| anyhow!("Falha ao gerar LLVM IR: {}", e))?;
        eprintln!("[Interceptor] LLVM IR gerado ({} bytes).", llvm_ir.len());

        // 6.4 Compilar LLVM IR para PTX
        let ptx_bytes = compile_to_ptx(&llvm_ir, &gpu.capabilities)
            .map_err(|e| anyhow!("Falha ao compilar para PTX: {}", e))?;
        eprintln!("[Interceptor] PTX compilado ({} bytes).", ptx_bytes.len());

        // 6.5 Salvar no cache L1
        let cached = local_cache::CachedKernel {
            binary: ptx_bytes.clone(),
            target: "ptx".to_string(),
        };
        self.local_cache.set(&cache_key, &cached);
        eprintln!("[Interceptor] Binário salvo no cache L1.");

        // 6.6 Opcional: salvar no Redis em background (não bloqueante)
        // tokio::spawn(async move { ClusterCache::set(&cache_key, &ptx_bytes).await });

        // ------------------------------------------------------------------
        // 7. Executar o kernel na GPU
        // ------------------------------------------------------------------
        self.execute_kernel(
            &ptx_bytes,
            &meta,
            device_ptr_x,
            device_ptr_y,
            n,
            Some(cache_key.clone()),
        )
    }

    // ----------------------------------------------------------------------
    // Função auxiliar para executar o kernel (cache hit ou miss)
    // ----------------------------------------------------------------------
    fn execute_kernel(
        &self,
        ptx_bytes: &[u8],
        meta: &KernelMetadata,
        device_ptr_x: *mut c_void,
        device_ptr_y: *mut c_void,
        n: i32,
        kernel_hash: Option<String>,
    ) -> Result<()> {
        // 1. Extrair parâmetros FLIR (tile_size, shared_mem_bytes)
        //    Como não temos acesso direto ao FlirOp aqui, reconstruímos a partir do meta.
        //    Em produção, você passaria o FlirOp inteiro.
        let tile_size = meta.capabilities.as_ref()
            .map(|c| c.max_threads_per_block)
            .unwrap_or(128);
        let shared_mem_bytes = (tile_size + 2) * 8; // radius=1, f64=8 bytes
        let flir_params = serde_json::json!({
            "tile_size": tile_size,
            "shared_mem_bytes": shared_mem_bytes,
        });

        // 2. Montar ponteiros de entrada/saída
        let input_ptrs = vec![device_ptr_x as *const c_void];
        let output_ptrs = vec![device_ptr_y as *const c_void];

        // 3. Executar via executor
        execute_flir_kernel(
            ptx_bytes,
            "basalto_kernel", // nome fixo gerado pelo flir_to_llvm
            &flir_params,
            &input_ptrs,
            &output_ptrs,
            n,
            kernel_hash.clone(),
            self.jit_sender.clone(),
        )
        .map_err(|e| anyhow!("Falha na execução do kernel: {}", e))?;

        // 4. Se auditoria estiver habilitada, gerar SHA-256 para telemetria
        if std::env::var("BASALTO_AUDIT_ENABLED").unwrap_or_default() == "true" {
            let audit_digest = meta.audit_digest();
            eprintln!("[Interceptor] Audit SHA-256: {}", audit_digest);
            // TODO: enviar para energy-telemetry com (audit_digest, timestamp, kWh)
        }

        eprintln!("[Interceptor] Kernel executado com sucesso.");
        Ok(())
    }
}

// --------------------------------------------------------------------------
// Bindings PyO3 – expõe a função ao Python
// --------------------------------------------------------------------------
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Wrapper Python para o interceptor.
#[pyclass]
pub struct PyBasaltoInterceptor {
    inner: BasaltoInterceptor,
}

#[pymethods]
impl PyBasaltoInterceptor {
    /// Cria uma nova instância (pode receber um canal opcional – ignoramos por simplicidade).
    #[new]
    pub fn new() -> Self {
        // Em produção, você passaria um canal para o JIT.
        let sender: Option<mpsc::Sender<KernelExecutionReport>> = None;
        let inner = BasaltoInterceptor::new(sender);
        Self { inner }
    }

    /// Função chamada pelo Python (via torch.compile)
    ///
    /// # Argumentos Python:
    /// - `op: str`
    /// - `dtype: str`
    /// - `shape: list[int]`
    /// - `job_id: str | None`
    /// - `device_ptr_x: int` (endereço do ponteiro CUDA, convertido para *mut c_void)
    /// - `device_ptr_y: int`
    /// - `n: int`
    ///
    /// Retorna `None` em caso de sucesso, ou levanta exceção em caso de erro.
    pub fn compile_and_execute(
        &self,
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

        self.inner.compile_and_execute(
            op,
            dtype,
            shape,
            job_id,
            ptr_x,
            ptr_y,
            n,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

/// Módulo Python (registrado no lib.rs)
#[pymodule]
pub fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBasaltoInterceptor>()?;
    Ok(())
}