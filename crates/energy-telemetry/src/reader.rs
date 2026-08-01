// crates/energy-telemetry/src/reader.rs
use anyhow::{anyhow, Result};
use std::process::Command;
use std::fs;
use std::path::Path;
use serde_json::Value;

// --------------------------------------------------------------------------
// Traço unificado para leitura de energia
// --------------------------------------------------------------------------
pub trait EnergyReader: Send + Sync {
    fn read_power_watts(&self) -> Result<f64>;      // potência instantânea (W)
    fn read_energy_joules(&self) -> Result<f64>;    // energia acumulada (J) – ideal para delta
}

// --------------------------------------------------------------------------
// Estrutura que gerencia a fonte ativa
// --------------------------------------------------------------------------
pub struct AutoEnergyReader {
    source: EnergySource,
    bmc_ip: Option<String>,
    redfish_user: Option<String>,
    redfish_pass: Option<String>,
}

#[derive(Clone, Copy, Debug)]
enum EnergySource {
    Redfish,
    Ipmi,
    Nvml,
    Unavailable,
}

impl AutoEnergyReader {
    /// Detecta automaticamente a melhor fonte disponível (com root)
    pub fn auto_detect() -> Self {
        // 1. Tenta Redfish (primeiro via SMBIOS, depois via config)
        if let Some(bmc_ip) = Self::discover_bmc_ip() {
            if Self::test_redfish_connection(&bmc_ip) {
                let (user, pass) = Self::load_redfish_credentials();
                eprintln!("[Telemetry] Redfish detectado em {}", bmc_ip);
                return Self {
                    source: EnergySource::Redfish,
                    bmc_ip: Some(bmc_ip),
                    redfish_user: user,
                    redfish_pass: pass,
                };
            }
        }

        // 2. Fallback para IPMI (verifica /dev/ipmi* e ipmitool)
        if Path::new("/dev/ipmi0").exists() || Path::new("/dev/ipmi/0").exists() {
            if Self::test_ipmi_connection() {
                eprintln!("[Telemetry] IPMI detectado via /dev/ipmi0");
                return Self {
                    source: EnergySource::Ipmi,
                    bmc_ip: None,
                    redfish_user: None,
                    redfish_pass: None,
                };
            }
        }

        // 3. Fallback para NVML (leitura por GPU)
        if Self::test_nvml() {
            eprintln!("[Telemetry] NVML (DCGM) detectado");
            return Self {
                source: EnergySource::Nvml,
                bmc_ip: None,
                redfish_user: None,
                redfish_pass: None,
            };
        }

        eprintln!("[Telemetry] Nenhuma fonte de telemetria de energia disponível.");
        Self {
            source: EnergySource::Unavailable,
            bmc_ip: None,
            redfish_user: None,
            redfish_pass: None,
        }
    }

    // ------------------------------------------------------------------
    // DETECÇÃO REDFISH
    // ------------------------------------------------------------------
    fn discover_bmc_ip() -> Option<String> {
        // Tenta ler via dmidecode (tipo 38 – IPMI Device Information)
        let output = Command::new("dmidecode")
            .args(["-t", "38"])
            .output()
            .ok()?;
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
        // Teste rápido: faz curl (ou reqwest) para /redfish/v1
        // Usamos `curl` por simplicidade; em produção, use `reqwest` com timeout curto.
        let url = format!("https://{}/redfish/v1", bmc_ip);
        let status = Command::new("curl")
            .args(["-k", "-s", "-o", "/dev/null", "-w", "%{http_code}", &url])
            .output()
            .ok()
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .and_then(|code| code.parse::<u16>().ok());
        matches!(status, Some(200) | Some(401) | Some(403)) // 401/403 significa que está vivo
    }

    fn load_redfish_credentials() -> (Option<String>, Option<String>) {
        // Lê /etc/basalto/redfish.conf (user, password)
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

    // ------------------------------------------------------------------
    // DETECÇÃO IPMI
    // ------------------------------------------------------------------
    fn test_ipmi_connection() -> bool {
        Command::new("ipmitool")
            .args(["mc", "info"])
            .output()
            .ok()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    // ------------------------------------------------------------------
    // DETECÇÃO NVML
    // ------------------------------------------------------------------
    fn test_nvml() -> bool {
        // Tenta carregar libnvidia-ml e inicializar
        unsafe {
            let lib = libloading::Library::new("libnvidia-ml.so.1").ok()?;
            let init: libloading::Symbol<unsafe extern "C" fn() -> u32> = lib.get(b"nvmlInit_v2").ok()?;
            let result = init();
            if result == 0 {
                // Para não vazar, devíamos chamar nvmlShutdown, mas para teste só retornamos true
                return true;
            }
            false
        }
    }
}

// ------------------------------------------------------------------
// IMPLEMENTAÇÃO DAS LEITURAS
// ------------------------------------------------------------------
impl EnergyReader for AutoEnergyReader {
    fn read_power_watts(&self) -> Result<f64> {
        match self.source {
            EnergySource::Redfish => self.read_redfish_power(),
            EnergySource::Ipmi => self.read_ipmi_power(),
            EnergySource::Nvml => self.read_nvml_power(),
            EnergySource::Unavailable => Err(anyhow!("Nenhuma fonte de telemetria disponível")),
        }
    }

    fn read_energy_joules(&self) -> Result<f64> {
        // Implementação similar, mas lendo contadores acumulados.
        // Para Redfish: GET /redfish/v1/Chassis/1/Power -> EnergyConsumedkWh (multiplica por 3.6e6)
        // Para IPMI: `ipmitool sdr list | grep -i energy` -> Energy Reading
        // Para NVML: não tem acumulado por GPU; precisamos integrar potência no tempo.
        todo!("Leitura de energia acumulada")
    }
}

// ------------------------------------------------------------------
// MÉTODOS PRIVADOS DE LEITURA
// ------------------------------------------------------------------
impl AutoEnergyReader {
    fn read_redfish_power(&self) -> Result<f64> {
        let ip = self.bmc_ip.as_ref().ok_or_else(|| anyhow!("BMC IP não definido"))?;
        let url = format!("https://{}/redfish/v1/Chassis/1/Power", ip);
        // Usa reqwest com certificados ignorados (self-signed).
        // Como temos root, podemos usar o certificado do sistema ou ignorar.
        let client = reqwest::blocking::Client::builder()
            .danger_accept_invalid_certs(true)
            .timeout(std::time::Duration::from_secs(2))
            .build()?;
        let resp = client.get(&url)
            .basic_auth(
                self.redfish_user.as_deref().unwrap_or("admin"),
                self.redfish_pass.as_deref().map(|p| p.as_str())
            )
            .send()?;
        let json: Value = resp.json()?;
        // Navega até PowerControl[0].PowerConsumedWatts
        let watts = json["PowerControl"][0]["PowerConsumedWatts"]
            .as_f64()
            .ok_or_else(|| anyhow!("Campo PowerConsumedWatts não encontrado"))?;
        Ok(watts)
    }

    fn read_ipmi_power(&self) -> Result<f64> {
        let output = Command::new("ipmitool")
            .args(["dcmi", "power", "reading"])
            .output()?;
        let stdout = String::from_utf8(output.stdout)?;
        for line in stdout.lines() {
            if line.contains("Instantaneous power") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(val) = parts.get(2) {
                    return Ok(val.parse::<f64>()?);
                }
            }
        }
        Err(anyhow!("Não foi possível extrair potência do IPMI"))
    }

    fn read_nvml_power(&self) -> Result<f64> {
        // Usa nvidia-smi como fallback rápido (já tem bindings)
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=power.draw", "--format=csv,noheader,nounits"])
            .output()?;
        let stdout = String::from_utf8(output.stdout)?;
        let first_line = stdout.lines().next().ok_or_else(|| anyhow!("Sem saída"))?;
        Ok(first_line.trim().parse::<f64>()? / 1.0) // já em Watts
    }
}