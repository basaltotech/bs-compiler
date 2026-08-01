use anyhow::Result;
use std::fs;
use basalto_common::permissions::ensure_root_or_die;

pub fn read_energy() -> Result<f64> {
    ensure_root_or_die()?;
    
    // Caminho padrão do driver NVIDIA/AMD para o sensor de potência no Linux
    let microwatts_str = fs::read_to_string("/sys/class/drm/card0/device/hwmon/hwmon0/power1_input")?;
    
    let microwatts: f64 = microwatts_str.trim().parse()?;
    
    // Converte microWatts para Watts ordinários (Ex: 350.0 W)
    Ok(microwatts / 1_000_000.0)
}
