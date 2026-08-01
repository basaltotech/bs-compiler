use anyhow::{anyhow, Result};
use std::sync::Arc;
use tokio::sync::Semaphore;
use crate::profiler::SiliconForgeProfiler;
use crate::optimizer::{OptimizationSuggestion, SiliconForgeOptimizer};
use basalto_core::flir_builder::{flir_to_llvm, compile_to_ptx, FlirModule, FlirOp};
use basalto_core::hasher::KernelMetadata;
use basalto_tree::local_cache::LocalCache;
use basalto_common::hardware::GpuIdentity;

pub struct SiliconForgeCompiler {
    profiler: Arc<SiliconForgeProfiler>,
    optimizer: Arc<SiliconForgeOptimizer>,
    local_cache: Arc<LocalCache>,
    gpu_identity: Arc<GpuIdentity>,
    semaphore: Arc<Semaphore>,
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
            semaphore: Arc::new(Semaphore::new(4)), // Máximo 4 compilações simultâneas
        }
    }

    pub async fn process_suggestion(&self, suggestion: OptimizationSuggestion) -> Result<()> {
        // Adquire permissão do semáforo para limitar compilações concorrentes
        let _permit = self.semaphore.acquire().await.unwrap();

        eprintln!(
            "[SiliconForge] Aplicando otimização: {} (confiança: {:.2})",
            suggestion.reason, suggestion.confidence
        );

        let hash = &suggestion.kernel_hash;

        // 1. Recuperar o cache completo (binário + metadados)
        let cached = self.local_cache
            .get(hash)
            .ok_or_else(|| anyhow!("Kernel não encontrado no cache: {}", hash))?;

        let original_meta = cached.metadata
            .ok_or_else(|| anyhow!("Metadados não disponíveis para este kernel (cache antigo)"))?;

        // 2. Aplicar as otimizações sugeridas
        let dtype = suggestion.new_precision.as_deref().unwrap_or(&original_meta.dtype);
        let dims = original_meta.shape.len();

        let coeffs = match dims {
            1 => vec![0.2, 0.3, 0.5],
            2 => vec![0.1, 0.2, 0.1, 0.2, 0.0, 0.2, 0.1, 0.2, 0.1],
            _ => vec![1.0 / 27.0; 27],
        };

        let new_tile_x = suggestion.new_tile_x.unwrap_or(original_meta.shape.get(0).copied().unwrap_or(128) as u32);
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

        // 3. Recompilar com os novos parâmetros
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

        // 4. Criar metadados atualizados (com novo dtype, tile sizes, etc.)
        let new_meta = KernelMetadata {
            operation: original_meta.operation.clone(),
            dtype: dtype.to_string(),
            shape: original_meta.shape.clone(),
            strides: original_meta.strides.clone(),
            vendor: original_meta.vendor.clone(),
            arch: original_meta.arch.clone(),
            driver_version: original_meta.driver_version.clone(),
            job_id: None,
            node_id: None,
            capabilities: original_meta.capabilities.clone(),
        };

        // 5. CORREÇÃO CRÍTICA: Sobrescrever a chave ORIGINAL
        //    O interceptor sempre consulta o cache com esta chave.
        //    Se salvarmos com uma chave nova, a otimização nunca será aplicada.
        let optimized_entry = basalto_tree::local_cache::CachedKernel {
            binary: ptx_bytes,
            target: "ptx".to_string(),
            tile_x: Some(new_tile_x),
            tile_y: Some(new_tile_y),
            shared_mem_bytes: new_shared_mem,
            radius: 1,
            metadata: Some(new_meta),
        };

        self.local_cache.set(hash, &optimized_entry);

        eprintln!(
            "[SiliconForge] Kernel otimizado SUBSTITUIU o original com sucesso (hash: {})",
            hash
        );

        Ok(())
    }
}