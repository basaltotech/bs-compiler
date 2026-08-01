use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub kernel_hash: String,
    pub iteration: u64,
    pub timestamp: u64,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u32,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
}

pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    max_checkpoints: usize,
    // Mantém um histórico dos hashes em ordem de criação
    history: VecDeque<String>,
}

impl CheckpointManager {
    pub fn new(checkpoint_dir: &str, max_checkpoints: usize) -> Self {
        let dir = PathBuf::from(checkpoint_dir);
        let _ = fs::create_dir_all(&dir);
        Self {
            checkpoint_dir: dir,
            max_checkpoints,
            history: VecDeque::with_capacity(max_checkpoints),
        }
    }

    pub fn save(
        &mut self,
        hash: &str,
        data: &[u8],
        metadata: CheckpointMetadata,
    ) -> Result<(), String> {
        let path = self.checkpoint_dir.join(format!("{}.ckpt", hash));
        fs::write(&path, data).map_err(|e| e.to_string())?;

        // Salva metadados separadamente
        let meta_path = self.checkpoint_dir.join(format!("{}.meta", hash));
        let meta_bytes = serde_json::to_vec(&metadata).map_err(|e| e.to_string())?;
        fs::write(meta_path, meta_bytes).map_err(|e| e.to_string())?;

        self.history.push_back(hash.to_string());

        // Rotacionar se exceder o limite
        while self.history.len() > self.max_checkpoints {
            if let Some(oldest) = self.history.pop_front() {
                let _ = fs::remove_file(self.checkpoint_dir.join(format!("{}.ckpt", oldest)));
                let _ = fs::remove_file(self.checkpoint_dir.join(format!("{}.meta", oldest)));
            }
        }

        Ok(())
    }

    pub fn load(&self, hash: &str) -> Result<Vec<u8>, String> {
        let path = self.checkpoint_dir.join(format!("{}.ckpt", hash));
        fs::read(&path).map_err(|e| e.to_string())
    }

    pub fn load_metadata(&self, hash: &str) -> Result<CheckpointMetadata, String> {
        let path = self.checkpoint_dir.join(format!("{}.meta", hash));
        let bytes = fs::read(&path).map_err(|e| e.to_string())?;
        serde_json::from_slice(&bytes).map_err(|e| e.to_string())
    }

    pub fn list_checkpoints(&self) -> Vec<String> {
        self.history.iter().cloned().collect()
    }

    pub fn has_checkpoint(&self, hash: &str) -> bool {
        self.checkpoint_dir.join(format!("{}.ckpt", hash)).exists()
    }

    pub fn get_latest(&self) -> Option<String> {
        self.history.back().cloned()
    }
}