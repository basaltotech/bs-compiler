use blake3;
use sha2::{Digest, Sha256};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelMetadata {
    pub operation: String,      // "matmul", "attention", "softmax"
    pub dtype: String,          // "f32", "bf16", "f16"
    pub shape: Vec<usize>,
    pub vendor: String,
    pub arch: String,
    pub driver_version: String,
    // Auditoria adicional
    pub job_id: Option<String>,
    pub node_id: Option<String>,
}

impl KernelMetadata {
    /// Serializa de forma determinística: tamanho fixo por campo para evitar ambiguidade.
    pub fn to_serialized(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // operation: 32 bytes fixos
        let mut op = [0u8; 32];
        let op_bytes = self.operation.as_bytes();
        let len = op_bytes.len().min(32);
        op[..len].copy_from_slice(&op_bytes[..len]);
        buf.extend_from_slice(&op);

        // dtype: 16 bytes
        let mut dt = [0u8; 16];
        let dt_bytes = self.dtype.as_bytes();
        let len = dt_bytes.len().min(16);
        dt[..len].copy_from_slice(&dt_bytes[..len]);
        buf.extend_from_slice(&dt);

        // shape: 8 bytes para len, depois cada dimensão em 8 bytes
        buf.extend_from_slice(&(self.shape.len() as u64).to_le_bytes());
        for dim in &self.shape {
            buf.extend_from_slice(&(*dim as u64).to_le_bytes());
        }

        // vendor (16), arch (16), driver_version (32)
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

        // job_id, node_id opcionais (se existirem)
        if let Some(job) = &self.job_id {
            let mut j = [0u8; 32];
            let j_bytes = job.as_bytes();
            let len = j_bytes.len().min(32);
            j[..len].copy_from_slice(&j_bytes[..len]);
            buf.extend_from_slice(&j);
        } else {
            buf.extend_from_slice(&[0u8; 32]);
        }
        if let Some(node) = &self.node_id {
            let mut n = [0u8; 32];
            let n_bytes = node.as_bytes();
            let len = n_bytes.len().min(32);
            n[..len].copy_from_slice(&n_bytes[..len]);
            buf.extend_from_slice(&n);
        } else {
            buf.extend_from_slice(&[0u8; 32]);
        }
        buf
    }

    /// Chave para cache (BLAKE3)
    pub fn cache_key(&self) -> String {
        let data = self.to_serialized();
        let hash = blake3::hash(&data);
        hash.to_hex().to_string()
    }

    /// Digest para auditoria (SHA-256)
    pub fn audit_digest(&self) -> String {
        let data = self.to_serialized();
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let result = hasher.finalize();
        hex::encode(result)
    }
}