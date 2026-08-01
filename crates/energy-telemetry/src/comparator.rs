use dashmap::DashMap;
use std::sync::Arc;
use std::collections::HashMap;
use crate::correlator::Correlator;

pub struct TemporalComparator {
    correlator: Arc<Correlator>,
    // Map: (operation, dtype, shape) -> Vec de hashes ordenados por tempo
    history: Arc<DashMap<String, Vec<String>>>,
}

impl TemporalComparator {
    pub fn new(correlator: Arc<Correlator>) -> Self {
        Self {
            correlator,
            history: Arc::new(DashMap::new()),
        }
    }

    fn build_key(op: &str, dtype: &str, shape: &[usize]) -> String {
        format!("{}:{}:{:?}", op, dtype, shape)
    }

    pub fn record_execution(
        &self,
        hash: &str,
        op: &str,
        dtype: &str,
        shape: &[usize],
        timestamp: u64,
    ) {
        let key = Self::build_key(op, dtype, shape);
        let mut entry = self.history.entry(key).or_insert_with(Vec::new);
        entry.push(hash.to_string());
        // Mantém ordenado por timestamp (assumindo que insert já está em ordem)
    }

    pub fn get_previous_execution(
        &self,
        op: &str,
        dtype: &str,
        shape: &[usize],
        current_hash: &str,
    ) -> Option<String> {
        let key = Self::build_key(op, dtype, shape);
        if let Some(entry) = self.history.get(&key) {
            let vec = entry.value();
            if let Some(pos) = vec.iter().position(|h| h == current_hash) {
                if pos > 0 {
                    return Some(vec[pos - 1].clone());
                }
            }
        }
        None
    }

    pub fn compute_delta(
        &self,
        current_hash: &str,
        previous_hash: &str,
    ) -> Option<(f64, f64, u64)> {
        let current = self.correlator.get(current_hash)?;
        let previous = self.correlator.get(previous_hash)?;
        let delta_kwh = current.2 - previous.2;
        let delta_duration = current.3 - previous.3;
        Some((delta_kwh, delta_kwh / previous.2, delta_duration))
    }
}