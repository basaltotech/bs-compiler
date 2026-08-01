use std::fs;
use std::process::Command;
use libloading::{Library, Symbol};
use serde::{Deserialize, Serialize};
use crate::error::BasaltoError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub max_threads_per_block: i32,
    pub max_shared_memory_per_block: u64,
    pub max_registers_per_block: i32,
    pub warp_size: i32,
    pub multi_processor_count: i32,
}

impl DeviceCapabilities {
    pub fn from_nvidia_device(device_index: i32) -> Option<Self> {
        unsafe {
            let lib = Library::new("libcuda.so.1").ok()?;
            type CuInit = unsafe extern "C" fn(u32) -> u32;
            type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> u32;
            type CuDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, i32) -> u32;

            let cu_init: Symbol<CuInit> = lib.get(b"cuInit\0").ok()?;
            let cu_device_get: Symbol<CuDeviceGet> = lib.get(b"cuDeviceGet\0").ok()?;
            let cu_device_get_attr: Symbol<CuDeviceGetAttribute> = lib.get(b"cuDeviceGetAttribute\0").ok()?;

            if cu_init(0) != 0 { return None; }
            let mut device = 0;
            if cu_device_get(&mut device, device_index) != 0 { return None; }

            const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
            const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
            const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 1;
            const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: i32 = 8;
            const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: i32 = 12;
            const CU_DEVICE_ATTRIBUTE_WARP_SIZE: i32 = 5;
            const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;

            let get_attr = |attr: i32| -> Option<i32> {
                let mut val = 0;
                if cu_device_get_attr(&mut val, attr, device) == 0 { Some(val) } else { None }
            };

            Some(DeviceCapabilities {
                compute_capability_major: get_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                compute_capability_minor: get_attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
                max_threads_per_block: get_attr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
                max_shared_memory_per_block: (get_attr(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)? as u64) * 1024,
                max_registers_per_block: get_attr(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)?,
                warp_size: get_attr(CU_DEVICE_ATTRIBUTE_WARP_SIZE)?,
                multi_processor_count: get_attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
            })
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuIdentity {
    pub vendor: String,
    pub arch: String,
    pub driver_version: String,
    pub node_id: String,
    pub capabilities: Option<DeviceCapabilities>,
}

impl GpuIdentity {
    pub fn from_system() -> Result<Self, BasaltoError> {
        let vendor = detect_vendor();
        let arch = detect_arch(&vendor);
        let driver_version = detect_driver_version(&vendor);
        let node_id = fs::read_to_string("/etc/hostname")
            .unwrap_or_else(|_| "unknown-node".to_string())
            .trim()
            .to_string();

        let capabilities = if vendor == "nvidia" {
            DeviceCapabilities::from_nvidia_device(0)
        } else {
            None
        };

        Ok(GpuIdentity { vendor, arch, driver_version, node_id, capabilities })
    }
}

fn detect_vendor() -> String {
    if fs::metadata("/proc/driver/nvidia/version").is_ok() { return "nvidia".to_string(); }
    if let Ok(entries) = fs::read_dir("/sys/class/drm/") {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path().join("device").join("vendor");
            if let Ok(content) = fs::read_to_string(&path) {
                let vendor_id = content.trim();
                if vendor_id == "0x1002" { return "amd".to_string(); }
                if vendor_id == "0x8086" { return "intel".to_string(); }
            }
        }
    }
    "unknown".to_string()
}

fn detect_arch(vendor: &str) -> String {
    match vendor {
        "nvidia" => {
            let output = Command::new("nvidia-smi")
                .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
                .output()
                .ok()?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let cleaned = stdout.trim().replace('.', "");
            if !cleaned.is_empty() { return format!("sm_{}", cleaned); }
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
                        if parts.len() >= 3 { return parts[2].to_string(); }
                    }
                }
            }
            "unknown".to_string()
        }
        _ => "unknown".to_string(),
    }
}