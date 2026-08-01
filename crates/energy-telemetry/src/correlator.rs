use dashmap::DashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct Correlator {
    records: Arc<DashMap<String, (String, String, f64, u64, u64)>>,
    // hash -> (job_id, node_id, kwh, duration_us, timestamp)
}

impl Correlator {
    pub fn new() -> Self {
        Self {
            records: Arc::new(DashMap::new()),
        }
    }

    pub fn record(
        &self,
        hash: &str,
        job_id: &str,
        node_id: &str,
        kwh: f64,
        duration_us: u64,
    ) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.records.insert(
            hash.to_string(),
            (
                job_id.to_string(),
                node_id.to_string(),
                kwh,
                duration_us,
                timestamp,
            ),
        );
    }

    pub fn get(&self, hash: &str) -> Option<(String, String, f64, u64, u64)> {
        self.records.get(hash).map(|v| v.clone())
    }

    pub fn get_all(&self) -> Vec<(String, String, String, f64, u64, u64)> {
        let mut result = Vec::new();
        for entry in self.records.iter() {
            let (hash, (job_id, node_id, kwh, duration_us, timestamp)) = entry.pair();
            result.push((
                hash.clone(),
                job_id.clone(),
                node_id.clone(),
                *kwh,
                *duration_us,
                *timestamp,
            ));
        }
        result
    }
}