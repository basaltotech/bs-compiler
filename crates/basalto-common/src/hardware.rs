// crates/basalto-common/src/hardware.rs
use std::fs;
use std::process::Command;
use libloading::{Library, Symbol};

// --------------------------------------------------------------------------
// 1. Estrutura que descreve as capacidades reais da GPU (leitura via root)
// --------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DeviceCapabilities {
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub max_threads_per_block: i32,
    pub max_shared_memory_per_block: u64,   // em bytes
    pub max_registers_per_block: i32,
    pub warp_size: i32,
    pub multi_processor_count: i32,
}

impl DeviceCapabilities {
    /// Lê todas as capacidades diretamente via CUDA Driver API.
    /// Requer acesso aos dispositivos (/dev/nvidia*) e à libcuda.so.1.
    /// Retorna `None` se falhar (ex: sem permissão, driver não carregado).
    pub fn from_nvidia_device(device_index: i32) -> Option<Self> {
        unsafe {
            // Carrega a libcuda dinamicamente
            let lib = match Library::new("libcuda.so.1") {
                Ok(l) => l,
                Err(_) => return None,
            };

            // Declara os tipos das funções que vamos usar
            type CuInit = unsafe extern "C" fn(u32) -> u32;
            type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> u32;
            type CuDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> u32;

            // Obtém os símbolos
            let cu_init: Symbol<CuInit> = match lib.get(b"cuInit\0") {
                Ok(s) => s,
                Err(_) => return None,
            };
            let cu_device_get: Symbol<CuDeviceGet> = match lib.get(b"cuDeviceGet\0") {
                Ok(s) => s,
                Err(_) => return None,
            };
            let cu_device_get_attr: Symbol<CuDeviceGetAttribute> = match lib.get(b"cuDeviceGetAttribute\0") {
                Ok(s) => s,
                Err(_) => return None,
            };

            // Inicializa o driver
            if cu_init(0) != 0 {
                return None;
            }

            // Obtém o dispositivo especificado
            let mut device = 0;
            if cu_device_get(&mut device, device_index) != 0 {
                return None;
            }

            // Constantes da CUDA Driver API (valores numéricos)
            const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
            const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
            const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 1;
            const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;   // retorna KB
            const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: i32 = 12;
            const CU_DEVICE_ATTRIBUTE_WARP_SIZE: i32 = 5;
            const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;

            // Função auxiliar para ler um atributo
            let get_attr = |attr: i32| -> Option<i32> {
                let mut val = 0;
                if cu_device_get_attr(&mut val, attr, device) == 0 {
                    Some(val)
                } else {
                    None
                }
            };

            let major = get_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
            let minor = get_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
            let max_threads = get_attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?;
            let shared_kb = get_attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?;
            let regs = get_attr(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)?;
            let warp = get_attr(CU_DEVICE_ATTRIBUTE_WARP_SIZE)?;
            let sm_count = get_attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;

            Some(DeviceCapabilities {
                compute_capability_major: major,
                compute_capability_minor: minor,
                max_threads_per_block: max_threads,
                max_shared_memory_per_block: (shared_kb as u64) * 1024,
                max_registers_per_block: regs,
                warp_size: warp,
                multi_processor_count: sm_count,
            })
        }
    }
}

// --------------------------------------------------------------------------
// 2. Estrutura principal de identidade da GPU (vendor, arch, driver, node)
// --------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GpuIdentity {
    pub vendor: String,
    pub arch: String,               // ex: "sm_80", "gfx90a", "pvc"
    pub driver_version: String,
    pub node_id: String,
    pub capabilities: Option<DeviceCapabilities>, // preenchido apenas se root e CUDA disponível
}

impl GpuIdentity {
    /// Coleta todas as informações do sistema.
    /// Tenta ler capacidades via CUDA API (root); se falhar, usa fallback com nvidia-smi.
    pub fn from_system() -> Self {
        let vendor = detect_vendor();
        let arch = detect_arch(&vendor);
        let driver_version = detect_driver_version(&vendor);
        let node_id = fs::read_to_string("/etc/hostname")
            .unwrap_or_else(|_| "unknown-node".to_string())
            .trim()
            .to_string();

        // Tenta ler capacidades reais (apenas NVIDIA, por enquanto)
        let capabilities = if vendor == "nvidia" {
            DeviceCapabilities::from_nvidia_device(0) // dispositivo 0
        } else {
            None
        };

        Self {
            vendor,
            arch,
            driver_version,
            node_id,
            capabilities,
        }
    }
}

// --------------------------------------------------------------------------
// 3. Funções auxiliares de detecção (vendors, arch, driver)
// --------------------------------------------------------------------------
fn detect_vendor() -> String {
    // NVIDIA: presença do /proc/driver/nvidia/version
    if fs::metadata("/proc/driver/nvidia/version").is_ok() {
        return "nvidia".to_string();
    }
    // AMD: verificar PCI vendor ID 0x1002 via sysfs
    // Exemplo: /sys/class/drm/card0/device/vendor -> 0x1002
    if let Ok(entries) = fs::read_dir("/sys/class/drm/") {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path().join("device").join("vendor");
            if let Ok(content) = fs::read_to_string(&path) {
                let vendor_id = content.trim();
                if vendor_id == "0x1002" {
                    return "amd".to_string();
                } else if vendor_id == "0x8086" {
                    return "intel".to_string();
                }
            }
        }
    }
    "unknown".to_string()
}

fn detect_arch(vendor: &str) -> String {
    match vendor {
        "nvidia" => {
            // Primeiro, se tivermos capabilities via CUDA, usamos isso
            // (já que a função from_system chama detect_arch antes de ler caps,
            //  precisamos fazer uma leitura separada para arch. Para simplificar,
            //  mantemos o fallback com nvidia-smi.)
            let output = Command::new("nvidia-smi")
                .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
                .output();
            if let Ok(out) = output {
                if let Ok(cc) = String::from_utf8(out.stdout) {
                    let cleaned = cc.trim().replace('.', "");
                    if !cleaned.is_empty() {
                        return format!("sm_{}", cleaned);
                    }
                }
            }
            // Fallback: se falhar, assumimos sm_70 (mais comum)
            "sm_70".to_string()
        }
        "amd" => "gfx90a".to_string(), // MI100/MI200
        "intel" => "pvc".to_string(),  // Ponte Vecchio
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
        // Para AMD/Intel, poderíamos ler via modinfo, mas deixamos unknown.
        _ => "unknown".to_string(),
    }
}

// --------------------------------------------------------------------------
// 4. Teste (ignorado por padrão, pois requer GPU real)
// --------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_hardware_detection() {
        let id = GpuIdentity::from_system();
        eprintln!("Vendor: {}", id.vendor);
        eprintln!("Arch: {}", id.arch);
        eprintln!("Driver: {}", id.driver_version);
        eprintln!("Node: {}", id.node_id);
        if let Some(caps) = id.capabilities {
            eprintln!("Capabilities: {:#?}", caps);
        } else {
            eprintln!("No capabilities (root or CUDA missing)");
        }
    }
}