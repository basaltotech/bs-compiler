use std::fs;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct CachedKernel {
    pub binary: Vec<u8>,
    pub target: String, // "ptx", "hsaco", "spirv"
}

pub struct LocalCache {
    root: PathBuf,
}

impl LocalCache {
    pub fn new() -> Self {
        let mut root = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
        root.push("basalto");
        root.push("kernels");
        fs::create_dir_all(&root).ok();
        Self { root }
    }

    pub fn get(&self, key: &str) -> Option<CachedKernel> {
        let path = self.root.join(key).with_extension("bin");
        if !path.exists() { return None; }
        let bytes = fs::read(&path).ok()?;
        bincode::deserialize(&bytes).ok()
    }

    pub fn set(&self, key: &str, kernel: &CachedKernel) {
        let path = self.root.join(key).with_extension("bin");
        let bytes = bincode::serialize(kernel).unwrap();
        fs::write(path, bytes).ok();
    }
}