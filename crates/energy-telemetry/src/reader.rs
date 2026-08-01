//! Leitura de telemetria de energia com suporte a NVML (NVIDIA), IPMI e Redfish.
//! Com acesso root, prioriza NVML para leitura precisa do contador de energia total.

use anyhow::{anyhow, Result};
use std::process::Command;
use std::fs;
use std::path::Path;
use std::ffi::c_void;
use libloading::{Library, Symbol};
use serde_json::Value;

// ============================================================================
// NVML bindings (carregados dinamicamente via libloading)
// ============================================================================
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

    /// Retorna o consumo total de energia da GPU 0 em milijoules (mJ).
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

    /// Retorna a potência instantânea em miliwatts (mW).
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

// ============================================================================
// Traço unificado para leitura de energia
// ============================================================================
pub trait EnergyReader: Send + Sync {
    fn read_power_watts(&self) -> Result<f64>;
    fn read_energy_joules(&self) -> Result<f64>;        // energia acumulada desde o boot (J)
    fn read_energy_delta_joules(&self, start_mj: u64) -> Result<f64>; // delta entre duas leituras
}

// ============================================================================
// Implementação principal com fallback automático
// ============================================================================
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
    /// Detecta automaticamente a melhor fonte disponível (prioriza NVML).
    pub fn auto_detect() -> Self {
        // 1. Tenta NVML (NVIDIA)
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

        // 2. Tenta IPMI
        if Path::new("/dev/ipmi0").exists() || Path::new("/dev/ipmi/0").exists() {
            if Self::test_ipmi_connection() {
                eprintln!("[Telemetry] IPMI detectado – usando como fallback.");
                return Self {
                    nvml: None,
                    source: EnergySource::Ipmi,
                    bmc_ip: None,
                    redfish_user: None,
                    redfish_pass: None,
                };
            }
        }

        // 3. Tenta Redfish (último fallback)
        if let Some(bmc_ip) = Self::discover_bmc_ip() {
            if Self::test_redfish_connection(&bmc_ip) {
                let (user, pass) = Self::load_redfish_credentials();
                eprintln!("[Telemetry] Redfish detectado em {} – usando como fallback.", bmc_ip);
                return Self {
                    nvml: None,
                    source: EnergySource::Redfish,
                    bmc_ip: Some(bmc_ip),
                    redfish_user: user,
                    redfish_pass: pass,
                };
            }
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
        // Tenta ler via dmidecode (tipo 38 – IPMI Device Information)
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
        // Alternativa: ler /etc/basalto/redfish.conf
        if let Ok(conf) = fs::read_to_string("/etc/basalto/redfish.conf") {
            for line in conf.lines() {
                if let Some(stripped) = line.strip_prefix("bmc_ip=") {
                    return Some(stripped.trim().to_string());
                }
            }
        }
        None
    }

    fn test_redfish_connection(bmc_ip: &str) -> bool {
        let url = format!("https://{}/redfish/v1", bmc_ip);
        let status = Command::new("curl")
            .args(["-k", "-s", "-o", "/dev/null", "-w", "%{http_code}", &url])
            .output()
            .ok()
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .and_then(|code| code.parse::<u16>().ok());
        matches!(status, Some(200) | Some(401) | Some(403))
    }

    fn load_redfish_credentials() -> (Option<String>, Option<String>) {
        if let Ok(conf) = fs::read_to_string("/etc/basalto/redfish.conf") {
            let mut user = None;
            let mut pass = None;
            for line in conf.lines() {
                if let Some(stripped) = line.strip_prefix("user=") {
                    user = Some(stripped.trim().to_string());
                }
                if let Some(stripped) = line.strip_prefix("password=") {
                    pass = Some(stripped.trim().to_string());
                }
            }
            return (user, pass);
        }
        (None, None)
    }

    fn test_ipmi_connection() -> bool {
        Command::new("ipmitool")
            .args(["mc", "info"])
            .output()
            .ok()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn read_ipmi_energy_joules() -> Result<f64> {
        let out = Command::new("ipmitool")
            .args(["sdr", "list"])
            .output()?;
        let s = String::from_utf8(out.stdout)?;
        for line in s.lines() {
            if line.contains("Energy") || line.contains("kWh") {
                if let Some(val) = line.split_whitespace().next() {
                    if let Ok(num) = val.parse::<f64>() {
                        return Ok(num * 3_600_000.0); // kWh → J
                    }
                }
            }
        }
        Err(anyhow!("Contador de energia IPMI não encontrado"))
    }

    fn read_redfish_energy_joules(bmc_ip: &str, user: Option<&str>, pass: Option<&str>) -> Result<f64> {
        let url = format!("https://{}/redfish/v1/Chassis/1/Power", bmc_ip);
        let client = reqwest::blocking::Client::builder()
            .danger_accept_invalid_certs(true)
            .timeout(std::time::Duration::from_secs(2))
            .build()?;
        let resp = client
            .get(&url)
            .basic_auth(user.unwrap_or("admin"), pass.map(|p| p.as_str()))
            .send()?;
        let json: Value = resp.json()?;
        let kwh = json["PowerControl"][0]["EnergyConsumedkWh"]
            .as_f64()
            .ok_or_else(|| anyhow!("EnergyConsumedkWh não encontrado"))?;
        Ok(kwh * 3_600_000.0) // kWh → J
    }
}

// ============================================================================
// Implementação da trait EnergyReader
// ============================================================================
impl EnergyReader for AutoEnergyReader {
    fn read_power_watts(&self) -> Result<f64> {
        match self.source {
            EnergySource::Nvml => {
                let nvml = self.nvml.as_ref().ok_or_else(|| anyhow!("NVML não disponível"))?;
                unsafe {
                    let mw = nvml.get_power_mw()?;
                    Ok(mw as f64 / 1000.0)
                }
            }
            EnergySource::Ipmi => {
                let out = Command::new("ipmitool")
                    .args(["dcmi", "power", "reading"])
                    .output()?;
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
                    .basic_auth(
                        self.redfish_user.as_deref().unwrap_or("admin"),
                        self.redfish_pass.as_deref().map(|p| p.as_str()),
                    )
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
                    Ok(mj as f64 / 1000.0) // mJ → J
                }
            }
            EnergySource::Ipmi => Self::read_ipmi_energy_joules(),
            EnergySource::Redfish => {
                let ip = self.bmc_ip.as_ref().ok_or_else(|| anyhow!("BMC IP não definido"))?;
                Self::read_redfish_energy_joules(ip, self.redfish_user.as_deref(), self.redfish_pass.as_deref())
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
                    Ok(delta_mj as f64 / 1000.0) // mJ → J
                }
            }
            _ => {
                // Fallback: lê energia total e subtrai
                let end_j = self.read_energy_joules()?;
                let start_j = start_mj as f64 / 1000.0;
                Ok(end_j - start_j)
            }
        }
    }
}