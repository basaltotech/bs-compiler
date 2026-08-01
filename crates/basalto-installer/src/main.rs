use anyhow::{anyhow, Result};
use basalto_common::hardware::GpuIdentity;
use basalto_common::permissions::{ensure_root_or_die, setup_udev_rules, add_user_to_group, detect_nvidia_group};
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    ensure_root_or_die()?;
    println!("[Installer] Detectando hardware...");
    let gpu = GpuIdentity::from_system()?;
    println!("Vendor: {}, Arch: {}", gpu.vendor, gpu.arch);

    if gpu.vendor == "nvidia" {
        setup_udev_rules()?;
        let group = detect_nvidia_group()?;
        add_user_to_group(&group)?;
        println!("Usuário adicionado ao grupo '{}' para acesso à GPU.", group);
    }

    let cache_dir = dirs::cache_dir().unwrap_or_else(|| std::path::PathBuf::from("/tmp")).join("basalto");
    fs::create_dir_all(&cache_dir)?;
    println!("Cache criado em {}", cache_dir.display());

    let conf = format!(
        "vendor={}\narch={}\ndriver={}\nnode_id={}\n",
        gpu.vendor, gpu.arch, gpu.driver_version, gpu.node_id
    );
    fs::write("/etc/basalto.conf", conf)?;

    let ld_conf_dir = "/etc/ld.so.conf.d/";
    fs::create_dir_all(ld_conf_dir)?;
    fs::write(format!("{}/cuda.conf", ld_conf_dir), "/usr/local/cuda/lib64\n")?;
    let status = std::process::Command::new("ldconfig")
        .status()
        .map_err(|e| anyhow!("Falha ao executar ldconfig: {}", e))?;
    if !status.success() {
        eprintln!("Aviso: ldconfig falhou (pode ser necessário reiniciar).");
    }

    let secret_path = "/etc/basalto/secret.key";
    if !Path::new(secret_path).exists() {
        use rand::RngCore;
        let mut key = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut key);
        fs::write(secret_path, &key)?;
        fs::set_permissions(secret_path, std::fs::Permissions::from_mode(0o600))?;
        println!("Chave secreta gerada em {}", secret_path);
    }

    Ok(())
}