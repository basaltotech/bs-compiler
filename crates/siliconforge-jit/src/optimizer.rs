use crate::profiler::KernelProfile;
use basalto_common::hardware::DeviceCapabilities;

#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    pub kernel_hash: String,
    pub new_tile_x: Option<u32>,
    pub new_tile_y: Option<u32>,
    pub new_shared_mem: Option<u32>,
    pub new_precision: Option<String>,
    pub reason: String,
    pub confidence: f32,
}

pub struct SiliconForgeOptimizer {
    caps: DeviceCapabilities,
    sensitivity: f32,
}

impl SiliconForgeOptimizer {
    pub fn new(caps: DeviceCapabilities) -> Self {
        Self {
            caps,
            sensitivity: 0.05,
        }
    }

    pub fn analyze(&self, profile: &KernelProfile) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        let hash = &profile.kernel_hash;

        if let Some(s) = self.suggest_block_optimization(profile) {
            suggestions.push(s);
        }
        if let Some(s) = self.suggest_shared_mem_optimization(profile) {
            suggestions.push(s);
        }
        if let Some(s) = self.suggest_precision_reduction(profile) {
            suggestions.push(s);
        }
        if let Some(s) = self.suggest_grid_optimization(profile) {
            suggestions.push(s);
        }
        suggestions
    }

    fn suggest_block_optimization(&self, profile: &KernelProfile) -> Option<OptimizationSuggestion> {
        let current_block_x = profile.avg_block.0 as u32;
        let current_block_y = profile.avg_block.1 as u32;
        let max_threads = self.caps.max_threads_per_block as u32;
        let current = current_block_x * current_block_y;
        let candidates = [128, 256, 512, 1024];

        for &candidate in &candidates {
            if candidate == current || candidate > max_threads {
                continue;
            }
            if profile.avg_duration_us > 1000.0 && candidate > current {
                return Some(OptimizationSuggestion {
                    kernel_hash: profile.kernel_hash.clone(),
                    new_tile_x: Some(candidate),
                    new_tile_y: Some(1),
                    new_shared_mem: None,
                    new_precision: None,
                    reason: format!(
                        "Aumentar block size de {} para {} pode reduzir latência (avg: {:.1}us)",
                        current, candidate, profile.avg_duration_us
                    ),
                    confidence: 0.7,
                });
            }
            if profile.avg_duration_us < 100.0 && candidate < current {
                return Some(OptimizationSuggestion {
                    kernel_hash: profile.kernel_hash.clone(),
                    new_tile_x: Some(candidate),
                    new_tile_y: Some(1),
                    new_shared_mem: None,
                    new_precision: None,
                    reason: format!(
                        "Diminuir block size de {} para {} pode melhorar ocupação (avg: {:.1}us)",
                        current, candidate, profile.avg_duration_us
                    ),
                    confidence: 0.6,
                });
            }
        }
        None
    }

    fn suggest_shared_mem_optimization(&self, profile: &KernelProfile) -> Option<OptimizationSuggestion> {
        let current_shared = profile.avg_shared_mem as u32;
        let max_shared = self.caps.max_shared_memory_per_block as u32;
        let occupancy = self.estimate_occupancy(profile);

        if occupancy < 0.5 && current_shared > 0 && current_shared < max_shared {
            let reduced = (current_shared as f32 * 0.75) as u32;
            if reduced > 1024 {
                return Some(OptimizationSuggestion {
                    kernel_hash: profile.kernel_hash.clone(),
                    new_tile_x: None,
                    new_tile_y: None,
                    new_shared_mem: Some(reduced),
                    new_precision: None,
                    reason: format!(
                        "Reduzir shared memory de {} para {} bytes pode melhorar ocupação (atual: {:.2})",
                        current_shared, reduced, occupancy
                    ),
                    confidence: 0.8,
                });
            }
        }
        None
    }

    fn suggest_precision_reduction(&self, profile: &KernelProfile) -> Option<OptimizationSuggestion> {
        if profile.avg_duration_us > 5000.0 {
            return Some(OptimizationSuggestion {
                kernel_hash: profile.kernel_hash.clone(),
                new_tile_x: None,
                new_tile_y: None,
                new_shared_mem: None,
                new_precision: Some("f16".to_string()),
                reason: format!(
                    "Kernel lento ({:.1}us) – tentar precisão mista (FP16) pode reduzir tempo",
                    profile.avg_duration_us
                ),
                confidence: 0.5,
            });
        }
        None
    }

    fn suggest_grid_optimization(&self, profile: &KernelProfile) -> Option<OptimizationSuggestion> {
        let grid_x = profile.avg_grid.0 as u32;
        let grid_y = profile.avg_grid.1 as u32;
        let sm_count = self.caps.multi_processor_count as u32;

        if grid_x * grid_y < sm_count * 2 && grid_x * grid_y > 0 {
            return Some(OptimizationSuggestion {
                kernel_hash: profile.kernel_hash.clone(),
                new_tile_x: None,
                new_tile_y: None,
                new_shared_mem: None,
                new_precision: None,
                reason: format!(
                    "Grid muito pequeno ({}) para {} SMs – considere reduzir block size",
                    grid_x * grid_y, sm_count
                ),
                confidence: 0.6,
            });
        }
        None
    }

    fn estimate_occupancy(&self, profile: &KernelProfile) -> f32 {
        let block_size = (profile.avg_block.0 * profile.avg_block.1) as u32;
        if block_size == 0 { return 0.0; }
        let max_threads = self.caps.max_threads_per_block as u32;
        let threads_per_sm = 1024;
        let blocks_per_sm = threads_per_sm / block_size.min(threads_per_sm);
        let sm_count = self.caps.multi_processor_count as u32;
        if sm_count == 0 { return 0.0; }
        let active_blocks = (profile.avg_grid.0 * profile.avg_grid.1) as f32 / sm_count as f32;
        (active_blocks / blocks_per_sm as f32).min(1.0)
    }
}