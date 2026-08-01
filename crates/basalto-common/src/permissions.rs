use std::fs;
use std::path::Path;
use std::process::Command;
use crate::error::BasaltoError;

pub fn ensure_root_or_die() -> Result<(), BasaltoError> {
    let uid = unsafe { libc::getuid() };
    if uid == 0 { return Ok(()); }
    if let Ok(caps) = fs::read_to_string("/proc/self/status") {
        for line in caps.lines() {
            if line.starts_with("CapEff:") {
                let cap_hex = line.split_whitespace().nth(1).unwrap_or("0");
                if let Ok(cap_val) = u64::from_str_radix(cap_hex, 16) {
                    if (cap_val >> 21) & 1 == 1 {
                        return Ok(());
                    }
                }
            }
        }
    }
    Err(BasaltoError::Permission("Root ou CAP_SYS_ADMIN necessário".to_string()))
}

pub fn setup_udev_rules() -> Result<(), BasaltoError> {
    let rule = r#"KERNEL=="nvidia*", MODE="0660", GROUP="video""#;
    fs::write("/etc/udev/rules.d/99-basalto-nvidia.rules", rule)
        .map_err(|e| BasaltoError::Permission(format!("Falha ao criar regra udev: {}", e)))?;
    Command::new("udevadm")
        .args(["control", "--reload-rules"])
        .output()
        .map_err(|e| BasaltoError::Permission(format!("Falha ao recarregar udev: {}", e)))?;
    Command::new("udevadm")
        .args(["trigger", "--type=subsystems", "--action=add", "/sys/class/nvidia"])
        .output()
        .map_err(|e| BasaltoError::Permission(format!("Falha ao ativar regra: {}", e)))?;
    Ok(())
}

pub fn add_user_to_group(group: &str) -> Result<(), BasaltoError> {
    let user = std::env::var("USER").unwrap_or_else(|_| "root".to_string());
    Command::new("usermod")
        .args(["-a", "-G", group, &user])
        .output()
        .map_err(|e| BasaltoError::Permission(format!("Falha ao adicionar ao grupo {}: {}", group, e)))?;
    Ok(())
}

/// Detecta o grupo correto (video ou render) que possui acesso ao dispositivo NVIDIA.
pub fn detect_nvidia_group() -> Result<String, BasaltoError> {
    if Path::new("/dev/nvidia0").exists() {
        use std::os::unix::fs::MetadataExt;
        let meta = fs::metadata("/dev/nvidia0")
            .map_err(|e| BasaltoError::Hardware(format!("Falha ao ler /dev/nvidia0: {}", e)))?;
        let gid = meta.gid();
        let group_name = fs::read_to_string("/etc/group")
            .ok()
            .and_then(|content| {
                content.lines().find_map(|line| {
                    let parts: Vec<&str> = line.split(':').collect();
                    if parts.len() >= 3 {
                        if let Ok(gid_val) = parts[2].parse::<u32>() {
                            if gid_val == gid {
                                return Some(parts[0].to_string());
                            }
                        }
                    }
                    None
                })
            })
            .unwrap_or_else(|| "video".to_string());
        Ok(group_name)
    } else {
        Ok("render".to_string())
    }
}