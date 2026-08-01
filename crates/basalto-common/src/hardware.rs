use std::fs;
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GpuIdentity {
    pub vendor: String,      // "nvidia", "amd", "intel"
    pub arch: String,        // "sm_80", "gfx90a", "pvc"
    pub driver_version: String,
    pub node_id: String,
}

impl GpuIdentity {
    /// Lê todas as informações do sistema. Requer root para /proc/driver/nvidia.
    /// Se não tiver root, retorna placeholders com fallback.
    pub fn from_system() -> Self {
        let vendor = detect_vendor();
        let arch = detect_arch(&vendor);
        let driver_version = detect_driver_version(&vendor);
        let node_id = fs::read_to_string("/etc/hostname")
            .unwrap_or_else(|_| "unknown-node".to_string())
            .trim()
            .to_string();
        Self { vendor, arch, driver_version, node_id }
    }
}

fn detect_vendor() -> String {
    if fs::metadata("/proc/driver/nvidia/version").is_ok() {
        return "nvidia".to_string();
    }
    // Checar /sys/class/drm/card*/device/vendor (0x1002 = AMD, 0x8086 = Intel)
    // Para simplificar, fallback para "unknown"
    "unknown".to_string()
}

fn detect_arch(vendor: &str) -> String {
    match vendor {
        "nvidia" => {
            // Tenta ler compute capability via nvidia-smi
            let output = Command::new("nvidia-smi")
                .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
                .output();
            if let Ok(out) = output {
                if let Ok(cc) = String::from_utf8(out.stdout) {
                    return format!("sm_{}", cc.trim().replace('.', ""));
                }
            }
            "sm_70".to_string()
        }
        "amd" => "gfx90a".to_string(),
        "intel" => "pvc".to_string(),
        _ => "generic".to_string(),
    }
}

fn detect_driver_version(vendor: &str) -> String {
    match vendor {
        "nvidia" => {
            if let Ok(content) = fs::read_to_string("/proc/driver/nvidia/version") {
                for line in content.lines() {
                    if line.contains("NVRM version") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 3 {
                            return parts[2].to_string();
                        }
                    }
                }
            }
            "unknown".to_string()
        }
        _ => "unknown".to_string(),
    }
}