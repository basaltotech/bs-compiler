use std::path::PathBuf;
use std::fs;
use std::sync::Mutex;
use lru::LruCache;
use serde::{Serialize, Deserialize};
use bincode;

#[derive(Clone, Serialize, Deserialize)]
pub struct CachedKernel {
    pub binary: Vec<u8>,
    pub target: String,
    pub tile_size: u32,
    pub shared_mem_bytes: u32,
    pub radius: u32,
}

pub struct LocalCache {
    root: PathBuf,
    in_memory: Mutex<LruCache<String, CachedKernel>>,
    capacity: usize,
}

impl LocalCache {
    pub fn new() -> Self {
        let root = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("/tmp")).join("basalto/kernels");
        fs::create_dir_all(&root).ok();
        Self {
            root,
            in_memory: Mutex::new(LruCache::new(10000.try_into().unwrap())),
            capacity: 10000,
        }
    }

    pub fn new_with_capacity(cap: usize) -> Self {
        let root = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("/tmp")).join("basalto/kernels");
        fs::create_dir_all(&root).ok();
        Self {
            root,
            in_memory: Mutex::new(LruCache::new(cap.try_into().unwrap())),
            capacity: cap,
        }
    }

    pub fn get(&self, key: &str) -> Option<CachedKernel> {
        // Tenta memória
        if let Some(cached) = self.in_memory.lock().unwrap().get(key) {
            return Some(cached.clone());
        }
        // Tenta disco
        let path = self.root.join(key).with_extension("bin");
        if path.exists() {
            if let Ok(bytes) = fs::read(&path) {
                if let Ok(cached) = bincode::deserialize(&bytes) {
                    let _ = self.in_memory.lock().unwrap().put(key.to_string(), cached.clone());
                    return Some(cached);
                }
            }
        }
        None
    }

    pub fn set(&self, key: &str, kernel: &CachedKernel) {
        // Guarda em memória
        self.in_memory.lock().unwrap().put(key.to_string(), kernel.clone());
        // Guarda em disco
        let path = self.root.join(key).with_extension("bin");
        if let Ok(bytes) = bincode::serialize(kernel) {
            let _ = fs::write(path, bytes);
        }
    }
}