use blake3;
use sha2::{Digest, Sha256};
use serde::{Serialize, Deserialize};
use basalto_common::hardware::{DeviceCapabilities};
use std::sync::OnceLock;

fn load_secret_key() -> [u8; 32] {
    let path = "/etc/basalto/secret.key";
    if let Ok(mut f) = std::fs::File::open(path) {
        use std::io::Read;
        let mut key = [0u8; 32];
        if f.read_exact(&mut key).is_ok() {
            return key;
        }
    }
    let mut key = [0u8; 32];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        use std::io::Read;
        f.read_exact(&mut key).ok();
    }
    let _ = std::fs::create_dir_all("/etc/basalto");
    let _ = std::fs::write(path, &key);
    key
}

fn secret_key() -> &'static [u8; 32] {
    static KEY: OnceLock<[u8; 32]> = OnceLock::new();
    KEY.get_or_init(load_secret_key)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelMetadata {
    pub operation: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub radius: usize,
    pub matmul_m: Option<usize>,
    pub matmul_n: Option<usize>,
    pub matmul_k: Option<usize>,
    pub matmul_trans_a: Option<bool>,
    pub matmul_trans_b: Option<bool>,
    pub matmul_batch: Option<usize>,
    pub vendor: String,
    pub arch: String,
    pub driver_version: String,
    pub job_id: Option<String>,
    pub node_id: Option<String>,
    pub capabilities: Option<DeviceCapabilities>,
}

impl KernelMetadata {
    fn to_serialized(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut op = [0u8; 32];
        let op_bytes = self.operation.as_bytes();
        let len = op_bytes.len().min(32);
        op[..len].copy_from_slice(&op_bytes[..len]);
        buf.extend_from_slice(&op);

        let mut dt = [0u8; 16];
        let dt_bytes = self.dtype.as_bytes();
        let len = dt_bytes.len().min(16);
        dt[..len].copy_from_slice(&dt_bytes[..len]);
        buf.extend_from_slice(&dt);

        buf.extend_from_slice(&(self.shape.len() as u64).to_le_bytes());
        for dim in &self.shape {
            buf.extend_from_slice(&(*dim as u64).to_le_bytes());
        }

        buf.extend_from_slice(&(self.strides.len() as u64).to_le_bytes());
        for stride in &self.strides {
            buf.extend_from_slice(&(*stride as i64).to_le_bytes());
        }

        buf.extend_from_slice(&(self.radius as u64).to_le_bytes());

        let has_matmul = self.matmul_m.is_some();
        buf.extend_from_slice(&[has_matmul as u8]);
        if has_matmul {
            buf.extend_from_slice(&self.matmul_m.unwrap_or(0).to_le_bytes());
            buf.extend_from_slice(&self.matmul_n.unwrap_or(0).to_le_bytes());
            buf.extend_from_slice(&self.matmul_k.unwrap_or(0).to_le_bytes());
            buf.extend_from_slice(&[self.matmul_trans_a.unwrap_or(false) as u8]);
            buf.extend_from_slice(&[self.matmul_trans_b.unwrap_or(false) as u8]);
            buf.extend_from_slice(&self.matmul_batch.unwrap_or(1).to_le_bytes());
        }

        let mut v = [0u8; 16];
        let v_bytes = self.vendor.as_bytes();
        let len = v_bytes.len().min(16);
        v[..len].copy_from_slice(&v_bytes[..len]);
        buf.extend_from_slice(&v);

        let mut a = [0u8; 16];
        let a_bytes = self.arch.as_bytes();
        let len = a_bytes.len().min(16);
        a[..len].copy_from_slice(&a_bytes[..len]);
        buf.extend_from_slice(&a);

        let mut dv = [0u8; 32];
        let dv_bytes = self.driver_version.as_bytes();
        let len = dv_bytes.len().min(32);
        dv[..len].copy_from_slice(&dv_bytes[..len]);
        buf.extend_from_slice(&dv);

        if let Some(caps) = &self.capabilities {
            buf.extend_from_slice(&caps.compute_capability_major.to_le_bytes());
            buf.extend_from_slice(&caps.compute_capability_minor.to_le_bytes());
            buf.extend_from_slice(&caps.max_threads_per_block.to_le_bytes());
            buf.extend_from_slice(&caps.max_shared_memory_per_block.to_le_bytes());
            buf.extend_from_slice(&caps.max_registers_per_block.to_le_bytes());
            buf.extend_from_slice(&caps.warp_size.to_le_bytes());
            buf.extend_from_slice(&caps.multi_processor_count.to_le_bytes());
        } else {
            buf.extend_from_slice(&[0u8; 7*8]);
        }

        buf
    }

    pub fn cache_key(&self) -> String {
        let data = self.to_serialized();
        let mut hasher = blake3::Hasher::new_keyed(secret_key());
        hasher.update(&data);
        hasher.finalize().to_hex().to_string()
    }

    pub fn audit_digest(&self) -> String {
        let mut data = self.to_serialized();
        if let Some(job) = &self.job_id {
            data.extend_from_slice(job.as_bytes());
        }
        if let Some(node) = &self.node_id {
            data.extend_from_slice(node.as_bytes());
        }
        let mut hasher = Sha256::new();
        hasher.update(&data);
        hex::encode(hasher.finalize())
    }
}