use anyhow::{anyhow, Result};
use std::path::Path;
use sysinfo::{Disks, Networks, Components, Users, System};

pub struct GpuInfo {
    pub vendor: String,
    pub arch: String,
    pub driver_version: String,
}

pub fn detect_gpu_dynamic() -> Result<GpuInfo> {
    // 1. Camada de Hardware / Driver Direct: Tenta encontrar assinaturas nos módulos do Kernel Linux
    // Supercomputadores usam Linux quase estritamente (Cray, Slurm clusters, etc.)
    
    // Teste para NVIDIA (Cuda / NVML)
    if Path::new("/proc/driver/nvidia/version").exists() {
        if let Ok(version_str) = std::fs::read_to_string("/proc/driver/nvidia/version") {
            // Extrai a versão do driver dinamicamente do procfs
            let version = version_str.lines().next()
                .and_then(|l| l.split("Kernel Module").nth(1))
                .unwrap_or("Unknown").trim().to_string();
            
            return Ok(GpuInfo {
                vendor: "NVIDIA".to_string(),
                arch: "sm_90".to_string(),
                driver_version: version,
            });
        }
    }

    // Teste para AMD (ROCm / HIP)
    if Path::new("/sys/class/kfd/kfd/topology/nodes").exists() || Path::new("/dev/kfd").exists() {
        // Detecção via sysfs do subsistema ROCK da AMD
        return Ok(GpuInfo {
            vendor: "AMD".to_string(),
            arch: "gfx942".to_string(), 
            driver_version: "ROCm Native".to_string(),
        });
    }

    // Teste para Intel (OneAPI / Level Zero)
    if Path::new("/sys/class/drm/card0/device/vendor").exists() {
        if let Ok(vendor_id) = std::fs::read_to_string("/sys/class/drm/card0/device/vendor") {
            if vendor_id.trim() == "0x8086" { // ID de Fornecedor PCI da Intel
                return Ok(GpuInfo {
                    vendor: "Intel".to_string(),
                    arch: "Xe-HPC".to_string(), // Linha de supercomputação Ponte Vecchio / Rialto
                    driver_version: "i915/xe".to_string(),
                });
            }
        }
    }

    let mut sys = System::new_all();
    sys.refresh_all();
    
    // Em supercomputadores baseados em nós heterogêneos, podemos ler os componentes do sistema
    // Nota: Para precisão cirúrgica de microarquitetura, o uso de bindgen com NVML (NVIDIA Management Library)
    // ou `rocm-smi` é o padrão ouro se o respectivo driver estiver presente.

    Err(anyhow!("Nenhum acelerador de HPC (NVIDIA, AMD, Intel) foi detectado no sistema."))
}
