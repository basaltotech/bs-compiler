use anyhow::Result;
use basalto_common::hardware::GpuIdentity;
use basalto_common::permissions::{ensure_root_or_die, setup_udev_rules, add_user_to_group};
use std::fs;

fn main() -> Result<()> {
    ensure_root_or_die()?;
    println!("[Installer] Detetando hardware...");
    let gpu = GpuIdentity::from_system()?;
    println!("Vendor: {}, Arch: {}", gpu.vendor, gpu.arch);

    if gpu.vendor == "nvidia" {
        setup_udev_rules()?;
        add_user_to_group("video")?;
        println!("Regras udev configuradas para NVIDIA.");
    }

    let cache_dir = dirs::cache_dir().unwrap_or_else(|| std::path::PathBuf::from("/tmp")).join("basalto");
    fs::create_dir_all(&cache_dir)?;
    println!("Cache criado em {}", cache_dir.display());

    // Escreve arquivo de configuração básico
    let conf = format!("vendor={}\narch={}\ndriver={}\n", gpu.vendor, gpu.arch, gpu.driver_version);
    fs::write("/etc/basalto.conf", conf)?;

    Ok(())
}