use anyhow::{Result, anyhow};
use std::process::Command;
use std::fs;
use std::path::Path;
use serde_json::Value;

pub trait EnergyReader {
    fn read_power_watts(&self) -> Result<f64>;
    fn read_energy_joules(&self) -> Result<f64>;
}

pub struct AutoEnergyReader {
    source: EnergySource,
    bmc_ip: Option<String>,
    redfish_user: Option<String>,
    redfish_pass: Option<String>,
}

#[derive(Clone, Copy, Debug)]
enum EnergySource { Redfish, Ipmi, Nvml, Unavailable }

impl AutoEnergyReader {
    pub fn auto_detect() -> Self {
        // ... (detecção igual à anterior) ...
        // Para simplificar, vamos retornar uma instância com fallback para NVML
        Self {
            source: EnergySource::Nvml,
            bmc_ip: None,
            redfish_user: None,
            redfish_pass: None,
        }
    }

    pub fn get_node_id(&self) -> String {
        fs::read_to_string("/etc/hostname").unwrap_or_else(|_| "unknown".to_string()).trim().to_string()
    }
}

impl EnergyReader for AutoEnergyReader {
    fn read_power_watts(&self) -> Result<f64> {
        match self.source {
            EnergySource::Nvml => {
                let out = Command::new("nvidia-smi")
                    .args(["--query-gpu=power.draw", "--format=csv,noheader,nounits"])
                    .output()?;
                let s = String::from_utf8(out.stdout)?;
                let first = s.lines().next().ok_or(anyhow!("Sem saída"))?;
                Ok(first.trim().parse::<f64>()?)
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
                Err(anyhow!("Não encontrado"))
            }
            _ => Err(anyhow!("Fonte não disponível")),
        }
    }

    fn read_energy_joules(&self) -> Result<f64> {
        // Medir energia acumulada: para NVML, não há contador direto;
        // podemos aproximar integrando potência, mas aqui usamos ipmitool se disponível.
        match self.source {
            EnergySource::Ipmi => {
                let out = Command::new("ipmitool")
                    .args(["sdr", "list"])
                    .output()?;
                let s = String::from_utf8(out.stdout)?;
                for line in s.lines() {
                    if line.contains("Energy") || line.contains("kWh") {
                        // Parse simplificado
                        if let Some(val) = line.split_whitespace().next() {
                            if let Ok(num) = val.parse::<f64>() {
                                return Ok(num * 3_600_000.0); // kWh -> J
                            }
                        }
                    }
                }
                Err(anyhow!("Contador de energia não encontrado"))
            }
            _ => Err(anyhow!("Fonte não suporta leitura de energia acumulada")),
        }
    }
}