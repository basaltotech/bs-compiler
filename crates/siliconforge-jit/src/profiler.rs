use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelExecutionRecord {
    pub kernel_hash: String,
    pub duration_us: u64,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u32,
    pub timestamp: u64,
    pub job_id: Option<String>,
    pub node_id: Option<String>,
    pub gpu_vendor: String,
    pub gpu_arch: String,
    pub driver_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelProfile {
    pub kernel_hash: String,
    pub execution_count: u64,
    pub avg_duration_us: f64,
    pub min_duration_us: u64,
    pub max_duration_us: u64,
    pub p50_duration_us: u64,
    pub p95_duration_us: u64,
    pub p99_duration_us: u64,
    pub avg_grid: (f32, f32, f32),
    pub avg_block: (f32, f32, f32),
    pub avg_shared_mem: f32,
    pub last_updated: u64,
}

pub struct SiliconForgeProfiler {
    records: Arc<DashMap<String, Vec<KernelExecutionRecord>>>,
    profiles: Arc<DashMap<String, KernelProfile>>,
    max_records_per_kernel: usize,
}

impl SiliconForgeProfiler {
    pub fn new() -> Self {
        Self {
            records: Arc::new(DashMap::new()),
            profiles: Arc::new(DashMap::new()),
            max_records_per_kernel: 1000,
        }
    }

    pub fn record(&self, record: KernelExecutionRecord) {
        let hash = record.kernel_hash.clone();
        let mut vec = self.records.entry(hash.clone()).or_insert_with(Vec::new);
        vec.push(record);
        if vec.len() > self.max_records_per_kernel {
            vec.drain(0..vec.len() - self.max_records_per_kernel);
        }
        self.update_profile(&hash);
    }

    fn update_profile(&self, hash: &str) {
        let records = match self.records.get(hash) {
            Some(r) => r,
            None => return,
        };
        if records.is_empty() {
            return;
        }
        let count = records.len() as u64;
        let total_duration: u64 = records.iter().map(|r| r.duration_us).sum();
        let avg = total_duration as f64 / count as f64;
        let min = records.iter().map(|r| r.duration_us).min().unwrap();
        let max = records.iter().map(|r| r.duration_us).max().unwrap();
        let mut durations: Vec<u64> = records.iter().map(|r| r.duration_us).collect();
        durations.sort_unstable();
        let p50 = durations[(count as usize / 2).min(durations.len() - 1)];
        let p95 = durations[((count as usize * 95) / 100).min(durations.len() - 1)];
        let p99 = durations[((count as usize * 99) / 100).min(durations.len() - 1)];

        let avg_grid = records.iter().fold((0.0, 0.0, 0.0), |acc, r| {
            (acc.0 + r.grid.0 as f32, acc.1 + r.grid.1 as f32, acc.2 + r.grid.2 as f32)
        });
        let avg_grid = (avg_grid.0 / count as f32, avg_grid.1 / count as f32, avg_grid.2 / count as f32);
        let avg_block = records.iter().fold((0.0, 0.0, 0.0), |acc, r| {
            (acc.0 + r.block.0 as f32, acc.1 + r.block.1 as f32, acc.2 + r.block.2 as f32)
        });
        let avg_block = (avg_block.0 / count as f32, avg_block.1 / count as f32, avg_block.2 / count as f32);
        let avg_shared_mem = records.iter().map(|r| r.shared_mem_bytes as f32).sum::<f32>() / count as f32;

        let profile = KernelProfile {
            kernel_hash: hash.to_string(),
            execution_count: count,
            avg_duration_us: avg,
            min_duration_us: min,
            max_duration_us: max,
            p50_duration_us: p50,
            p95_duration_us: p95,
            p99_duration_us: p99,
            avg_grid,
            avg_block,
            avg_shared_mem,
            last_updated: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        self.profiles.insert(hash.to_string(), profile);
    }

    pub fn get_profile(&self, hash: &str) -> Option<KernelProfile> {
        self.profiles.get(hash).map(|p| p.clone())
    }

    pub fn get_all_profiles(&self) -> Vec<KernelProfile> {
        self.profiles.iter().map(|entry| entry.value().clone()).collect()
    }

    pub fn get_recent_records(&self, hash: &str, n: usize) -> Vec<KernelExecutionRecord> {
        self.records
            .get(hash)
            .map(|vec| vec.iter().rev().take(n).cloned().collect())
            .unwrap_or_default()
    }
}