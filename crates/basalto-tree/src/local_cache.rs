use std::collections::HashMap;
use std::sync::Mutex;
use anyhow::Result;

static CACHE: Mutex<HashMap<String, Vec<u8>>> = Mutex::new(HashMap::new());

pub fn get(hash: &str) -> Option<Vec<u8>> {
    let cache = CACHE.lock().unwrap();
    cache.get(hash).cloned()
}

pub fn put(hash: &str, binary: &[u8]) {
    let mut cache = CACHE.lock().unwrap();
    cache.insert(hash.to_string(), binary.to_vec());
}