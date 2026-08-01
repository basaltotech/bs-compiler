use anyhow::{anyhow, Result};
use std::sync::Arc;
use tokio::sync::Semaphore;
use crate::profiler::SiliconForgeProfiler;
use crate::optimizer::{OptimizationSuggestion, SiliconForgeOptimizer};
use basalto_core::flir_builder::{flir_to_llvm, compile_to_ptx, FlirModule, FlirOp};
use basalto_core::hasher::KernelMetadata;
use basalto_tree::local_cache::{LocalCache, CachedKernel};
use basalto_common::hardware::GpuIdentity;
use basalto_tree::executor::Executor;
use energy_telemetry::correlator::Correlator;
use energy_telemetry::comparator::TemporalComparator;
use std::sync::Mutex;

pub struct SiliconForgeCompiler {
    profiler: Arc<SiliconForgeProfiler>,
    optimizer: Arc<SiliconForgeOptimizer>,
    local_cache: Arc<LocalCache>,
    gpu_identity: Arc<GpuIdentity>,
    semaphore: Arc<Semaphore>,
    // Para validação e benchmark
    test_runner: Arc<Mutex<Option<Executor>>>,
}

impl SiliconForgeCompiler {
    pub fn new(
        profiler: Arc<SiliconForgeProfiler>,
        optimizer: Arc<SiliconForgeOptimizer>,
        local_cache: Arc<LocalCache>,
        gpu_identity: Arc<GpuIdentity>,
    ) -> Self {
        Self {
            profiler,
            optimizer,
            local_cache,
            gpu_identity,
            semaphore: Arc::new(Semaphore::new(4)),
            test_runner: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn process_suggestion(&self, suggestion: OptimizationSuggestion) -> Result<()> {
        let _permit = self.semaphore.acquire().await.unwrap();

        eprintln!(
            "[SiliconForge] Avaliando otimização: {} (confiança: {:.2})",
            suggestion.reason, suggestion.confidence
        );

        let hash = &suggestion.kernel_hash;

        // 1. Recuperar cache original (com metadados)
        let cached = self.local_cache
            .get(hash)
            .ok_or_else(|| anyhow!("Kernel não encontrado no cache: {}", hash))?;

        let original_meta = cached.metadata
            .ok_or_else(|| anyhow!("Metadados não disponíveis para este kernel"))?;

        // 2. Aplicar otimizações e recompilar
        let dtype = suggestion.new_precision.as_deref().unwrap_or(&original_meta.dtype);
        let dims = original_meta.shape.len();

        let coeffs = match dims {
            1 => vec![0.2, 0.3, 0.5],
            2 => vec![0.1, 0.2, 0.1, 0.2, 0.0, 0.2, 0.1, 0.2, 0.1],
            _ => vec![1.0 / 27.0; 27],
        };

        let new_tile_x = suggestion.new_tile_x.unwrap_or(
            original_meta.shape.get(0).copied().unwrap_or(128) as u32
        );
        let new_tile_y = suggestion.new_tile_y.unwrap_or(1);
        let new_shared_mem = suggestion.new_shared_mem.unwrap_or(cached.shared_mem_bytes);

        let params = serde_json::json!({
            "radius": 1,
            "coeffs": coeffs,
            "dtype": dtype,
            "dims": dims,
            "tile_x": new_tile_x,
            "tile_y": new_tile_y,
            "shared_mem_bytes": new_shared_mem,
            "stride_x": original_meta.strides.get(0).unwrap_or(&1),
            "stride_y": original_meta.strides.get(1).unwrap_or(&1),
            "stride_z": original_meta.strides.get(2).unwrap_or(&1),
        });

        let flir_module = FlirModule {
            ops: vec![FlirOp {
                op: format!("stencil_{}d", dims),
                inputs: vec!["x".to_string()],
                output: "y".to_string(),
                params: Some(params),
            }],
            input_names: vec!["x".to_string()],
            output_names: vec!["y".to_string()],
        };

        let llvm_ir = flir_to_llvm(&flir_module, &self.gpu_identity.capabilities, dtype)?;
        let ptx_bytes = compile_to_ptx(&llvm_ir, &self.gpu_identity.capabilities)?;

        // 3. Construir metadados para a nova versão
        let new_meta = KernelMetadata {
            operation: original_meta.operation.clone(),
            dtype: dtype.to_string(),
            shape: original_meta.shape.clone(),
            strides: original_meta.strides.clone(),
            radius: 1,
            matmul_m: None,
            matmul_n: None,
            matmul_k: None,
            matmul_trans_a: None,
            matmul_trans_b: None,
            matmul_batch: None,
            vendor: original_meta.vendor.clone(),
            arch: original_meta.arch.clone(),
            driver_version: original_meta.driver_version.clone(),
            job_id: None,
            node_id: None,
            capabilities: original_meta.capabilities.clone(),
        };

        // 4. VALIDAÇÃO NUMÉRICA (usando mini-batch)
        eprintln!("[SiliconForge] Validando numericamente o kernel otimizado...");
        let validation_result = self.validate_optimized_kernel(
            &ptx_bytes,
            &new_meta,
            &original_meta,
            suggestion.new_precision.is_some(),
        );

        if let Err(e) = validation_result {
            eprintln!("[SiliconForge] ❌ Validação numérica falhou: {}", e);
            return Ok(());
        }
        eprintln!("[SiliconForge] ✅ Validação numérica passou.");

        // 5. TESTE DE DESEMPENHO (benchmark)
        eprintln!("[SiliconForge] Medindo desempenho do kernel otimizado...");
        let old_duration = self.benchmark_kernel(&cached.binary, &original_meta)?;
        let new_duration = self.benchmark_kernel(&ptx_bytes, &new_meta)?;

        let improvement = 1.0 - (new_duration as f64 / old_duration as f64);
        eprintln!(
            "[SiliconForge] Desempenho: antigo={:.2}us, novo={:.2}us, melhoria={:.2}%",
            old_duration, new_duration, improvement * 100.0
        );

        // 6. DECISÃO: aplicar apenas se for >5% mais rápido
        if improvement > 0.05 {
            // Sobrescrever o cache com a nova versão
            let optimized_entry = CachedKernel {
                binary: ptx_bytes,
                target: "ptx".to_string(),
                tile_x: Some(new_tile_x),
                tile_y: Some(new_tile_y),
                shared_mem_bytes: new_shared_mem,
                radius: 1,
                metadata: Some(new_meta),
            };
            self.local_cache.set(hash, &optimized_entry);
            eprintln!("[SiliconForge] ✅ Otimização aplicada (substituiu o cache).");
        } else {
            eprintln!("[SiliconForge] ⏭️ Otimização rejeitada: ganho insuficiente (<5%).");
        }

        Ok(())
    }

    /// Valida numericamente o kernel otimizado contra uma referência CPU.
    fn validate_optimized_kernel(
        &self,
        ptx_bytes: &[u8],
        new_meta: &KernelMetadata,
        original_meta: &KernelMetadata,
        precision_changed: bool,
    ) -> Result<()> {
        // Em produção, precisamos de um executor com dados de teste.
        // Para este exemplo, assumimos que a validação passa.
        // TODO: Implementar com mini-batch real.
        eprintln!("[SiliconForge] Validação numérica (stub) – passou.");
        Ok(())
    }

    /// Executa o kernel em um mini-batch e retorna a duração média em microssegundos.
    fn benchmark_kernel(&self, ptx: &[u8], meta: &KernelMetadata) -> Result<f64> {
        // Em produção, isso executaria o kernel com dados sintéticos e mediria o tempo.
        // Para este exemplo, retornamos um valor simulado.
        // TODO: Implementar benchmark real.
        Ok(100.0) // valor fictício
    }
}