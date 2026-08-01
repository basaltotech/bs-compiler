use anyhow::{anyhow, Result};
use std::process::Command;
use std::fs;
use std::path::Path;
use std::ffi::c_void;
use libloading::{Library, Symbol};
use serde_json::Value;

type nvmlReturn_t = u32;
type nvmlDevice_t = *mut c_void;
const NVML_SUCCESS: nvmlReturn_t = 0;

pub struct NvmlRuntime {
    _lib: Library,
    pub nvml_init: Symbol<unsafe extern "C" fn() -> nvmlReturn_t>,
    pub nvml_shutdown: Symbol<unsafe extern "C" fn() -> nvmlReturn_t>,
    pub nvml_device_get_handle_by_index: Symbol<unsafe extern "C" fn(u32, *mut nvmlDevice_t) -> nvmlReturn_t>,
    pub nvml_device_get_total_energy_consumption: Symbol<unsafe extern "C" fn(nvmlDevice_t, *mut u64) -> nvmlReturn_t>,
    pub nvml_device_get_power_usage: Symbol<unsafe extern "C" fn(nvmlDevice_t, *mut u32) -> nvmlReturn_t>,
}

impl NvmlRuntime {
    pub fn new() -> Result<Self> {
        unsafe {
            let lib = Library::new("libnvidia-ml.so.1")
                .or_else(|_| Library::new("libnvidia-ml.so"))
                .map_err(|e| anyhow!("Falha ao carregar libnvidia-ml: {}", e))?;

            let nvml_init = lib.get(b"nvmlInit_v2\0")
                .map_err(|e| anyhow!("nvmlInit_v2 não encontrado: {}", e))?;
            let nvml_shutdown = lib.get(b"nvmlShutdown\0")
                .map_err(|e| anyhow!("nvmlShutdown não encontrado: {}", e))?;
            let nvml_device_get_handle_by_index = lib.get(b"nvmlDeviceGetHandleByIndex_v2\0")
                .map_err(|e| anyhow!("nvmlDeviceGetHandleByIndex_v2 não encontrado: {}", e))?;
            let nvml_device_get_total_energy_consumption = lib.get(b"nvmlDeviceGetTotalEnergyConsumption\0")
                .map_err(|e| anyhow!("nvmlDeviceGetTotalEnergyConsumption não encontrado: {}", e))?;
            let nvml_device_get_power_usage = lib.get(b"nvmlDeviceGetPowerUsage\0")
                .map_err(|e| anyhow!("nvmlDeviceGetPowerUsage não encontrado: {}", e))?;

            let ret = nvml_init();
            if ret != NVML_SUCCESS {
                return Err(anyhow!("nvmlInit falhou com código {}", ret));
            }

            Ok(Self {
                _lib: lib,
                nvml_init,
                nvml_shutdown,
                nvml_device_get_handle_by_index,
                nvml_device_get_total_energy_consumption,
                nvml_device_get_power_usage,
            })
        }
    }

    pub unsafe fn get_total_energy_mj(&self) -> Result<u64> {
        let mut device: nvmlDevice_t = std::ptr::null_mut();
        let ret = (self.nvml_device_get_handle_by_index)(0, &mut device);
        if ret != NVML_SUCCESS {
            return Err(anyhow!("nvmlDeviceGetHandleByIndex falhou com código {}", ret));
        }
        let mut energy_mj = 0u64;
        let ret = (self.nvml_device_get_total_energy_consumption)(device, &mut energy_mj);
        if ret != NVML_SUCCESS {
            return Err(anyhow!("nvmlDeviceGetTotalEnergyConsumption falhou com código {}", ret));
        }
        Ok(energy_mj)
    }

    pub unsafe fn get_power_mw(&self) -> Result<u32> {
        let mut device: nvmlDevice_t = std::ptr::null_mut();
        let ret = (self.nvml_device_get_handle_by_index)(0, &mut device);
        if ret != NVML_SUCCESS {
            return Err(anyhow!("nvmlDeviceGetHandleByIndex falhou com código {}", ret));
        }
        let mut power_mw = 0u32;
        let ret = (self.nvml_device_get_power_usage)(device, &mut power_mw);
        if ret != NVML_SUCCESS {
            return Err(anyhow!("nvmlDeviceGetPowerUsage falhou com código {}", ret));
        }
        Ok(power_mw)
    }
}

impl Drop for NvmlRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.nvml_shutdown)();
        }
    }
}

pub trait EnergyReader: Send + Sync {
    fn read_power_watts(&self) -> Result<f64>;
    fn read_energy_joules(&self) -> Result<f64>;
    fn read_energy_delta_joules(&self, start_mj: u64) -> Result<f64>;
    fn get_nvml(&self) -> Option<&NvmlRuntime>;
}

pub struct AutoEnergyReader {
    nvml: Option<NvmlRuntime>,
    source: EnergySource,
    bmc_ip: Option<String>,
    redfish_user: Option<String>,
    redfish_pass: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EnergySource {
    Nvml,
    Ipmi,
    Redfish,
    Unavailable,
}

impl AutoEnergyReader {
    pub fn auto_detect() -> Self {
        if let Ok(nvml) = NvmlRuntime::new() {
            eprintln!("[Telemetry] NVML detectado – usando para medição de energia.");
            return Self {
                nvml: Some(nvml),
                source: EnergySource::Nvml,
                bmc_ip: None,
                redfish_user: None,
                redfish_pass: None,
            };
        }

        if Path::new("/dev/ipmi0").exists() {
            eprintln!("[Telemetry] IPMI detectado – usando como fallback.");
            return Self {
                nvml: None,
                source: EnergySource::Ipmi,
                bmc_ip: None,
                redfish_user: None,
                redfish_pass: None,
            };
        }

        if let Some(bmc_ip) = Self::discover_bmc_ip() {
            eprintln!("[Telemetry] Redfish detectado em {} – usando como fallback.", bmc_ip);
            return Self {
                nvml: None,
                source: EnergySource::Redfish,
                bmc_ip: Some(bmc_ip),
                redfish_user: Some("admin".to_string()),
                redfish_pass: None,
            };
        }

        eprintln!("[Telemetry] Nenhuma fonte de telemetria de energia disponível.");
        Self {
            nvml: None,
            source: EnergySource::Unavailable,
            bmc_ip: None,
            redfish_user: None,
            redfish_pass: None,
        }
    }

    fn discover_bmc_ip() -> Option<String> {
        let output = Command::new("dmidecode").args(["-t", "38"]).output().ok()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.contains("IPMI Address") || line.contains("Base Address") {
                if let Some(ip) = line.split(':').nth(1) {
                    let ip = ip.trim();
                    if !ip.is_empty() && ip.chars().filter(|c| *c == '.').count() == 3 {
                        return Some(ip.to_string());
                    }
                }
            }
        }
        if let Ok(conf) = fs::read_to_string("/etc/basalto/redfish.conf") {
            for line in conf.lines() {
                if let Some(stripped) = line.strip_prefix("bmc_ip=") {
                    return Some(stripped.trim().to_string());
                }
            }
        }
        None
    }

    fn read_ipmi_energy_joules() -> Result<f64> {
        let out = Command::new("ipmitool").args(["sdr", "list"]).output()?;
        let s = String::from_utf8(out.stdout)?;
        for line in s.lines() {
            if line.contains("Energy") || line.contains("kWh") {
                if let Some(val) = line.split_whitespace().next() {
                    if let Ok(num) = val.parse::<f64>() {
                        return Ok(num * 3_600_000.0);
                    }
                }
            }
        }
        Err(anyhow!("Contador de energia IPMI não encontrado"))
    }
}

impl EnergyReader for AutoEnergyReader {
    fn get_nvml(&self) -> Option<&NvmlRuntime> {
        self.nvml.as_ref()
    }

    fn read_power_watts(&self) -> Result<f64> {
        match self.source {
            EnergySource::Nvml => {
                let nvml = self.nvml.as_ref().ok_or_else(|| anyhow!("NVML não disponível"))?;
                unsafe { Ok(nvml.get_power_mw()? as f64 / 1000.0) }
            }
            EnergySource::Ipmi => {
                let out = Command::new("ipmitool").args(["dcmi", "power", "reading"]).output()?;
                let s = String::from_utf8(out.stdout)?;
                for line in s.lines() {
                    if line.contains("Instantaneous power") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if let Some(val) = parts.get(2) {
                            return Ok(val.parse::<f64>()?);
                        }
                    }
                }
                Err(anyhow!("Potência IPMI não encontrada"))
            }
            EnergySource::Redfish => {
                let ip = self.bmc_ip.as_ref().ok_or_else(|| anyhow!("BMC IP não definido"))?;
                let url = format!("https://{}/redfish/v1/Chassis/1/Power", ip);
                let client = reqwest::blocking::Client::builder()
                    .danger_accept_invalid_certs(true)
                    .timeout(std::time::Duration::from_secs(2))
                    .build()?;
                let resp = client
                    .get(&url)
                    .basic_auth(self.redfish_user.as_deref().unwrap_or("admin"), self.redfish_pass.as_deref())
                    .send()?;
                let json: Value = resp.json()?;
                let watts = json["PowerControl"][0]["PowerConsumedWatts"]
                    .as_f64()
                    .ok_or_else(|| anyhow!("PowerConsumedWatts não encontrado"))?;
                Ok(watts)
            }
            EnergySource::Unavailable => Err(anyhow!("Nenhuma fonte de telemetria disponível")),
        }
    }

    fn read_energy_joules(&self) -> Result<f64> {
        match self.source {
            EnergySource::Nvml => {
                let nvml = self.nvml.as_ref().ok_or_else(|| anyhow!("NVML não disponível"))?;
                unsafe {
                    let mj = nvml.get_total_energy_mj()?;
                    Ok(mj as f64 / 1000.0)
                }
            }
            EnergySource::Ipmi => Self::read_ipmi_energy_joules(),
            EnergySource::Redfish => {
                let ip = self.bmc_ip.as_ref().ok_or_else(|| anyhow!("BMC IP não definido"))?;
                let url = format!("https://{}/redfish/v1/Chassis/1/Power", ip);
                let client = reqwest::blocking::Client::builder()
                    .danger_accept_invalid_certs(true)
                    .timeout(std::time::Duration::from_secs(2))
                    .build()?;
                let resp = client
                    .get(&url)
                    .basic_auth(self.redfish_user.as_deref().unwrap_or("admin"), self.redfish_pass.as_deref())
                    .send()?;
                let json: Value = resp.json()?;
                let kwh = json["PowerControl"][0]["EnergyConsumedkWh"]
                    .as_f64()
                    .ok_or_else(|| anyhow!("EnergyConsumedkWh não encontrado"))?;
                Ok(kwh * 3_600_000.0)
            }
            EnergySource::Unavailable => Err(anyhow!("Nenhuma fonte de telemetria disponível")),
        }
    }

    fn read_energy_delta_joules(&self, start_mj: u64) -> Result<f64> {
        match self.source {
            EnergySource::Nvml => {
                let nvml = self.nvml.as_ref().ok_or_else(|| anyhow!("NVML não disponível"))?;
                unsafe {
                    let end_mj = nvml.get_total_energy_mj()?;
                    let delta_mj = end_mj - start_mj;
                    Ok(delta_mj as f64 / 1000.0)
                }
            }
            _ => {
                let end_j = self.read_energy_joules()?;
                let start_j = start_mj as f64 / 1000.0;
                Ok(end_j - start_j)
            }
        }
    }
}