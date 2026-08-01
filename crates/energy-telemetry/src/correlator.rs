use dashmap::DashMap;
use std::sync::Arc;

pub struct Correlator {
    records: Arc<DashMap<String, (String, String, f64, u64)>>, // hash -> (job_id, node_id, kwh, duration_us)
}

impl Correlator {
    pub fn new() -> Self {
        Self { records: Arc::new(DashMap::new()) }
    }

    pub fn record(&self, hash: &str, job_id: &str, node_id: &str, kwh: f64, duration_us: u64) {
        self.records.insert(hash.to_string(), (job_id.to_string(), node_id.to_string(), kwh, duration_us));
    }

    pub fn get(&self, hash: &str) -> Option<(String, String, f64, u64)> {
        self.records.get(hash).map(|v| v.clone())
    }
}