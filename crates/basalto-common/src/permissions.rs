use anyhow::{bail, Result};
use std::os::unix::fs::PermissionsExt;

pub fn ensure_root_or_die() -> Result<()> {
    if unsafe { libc::getuid() } != 0 {
        bail!("Basalto exige root (sudo).");
    }
    Ok(())
}

pub fn check_device_node(path: &str) -> Result<bool> {
    ensure_root_or_die()?;
    Ok(true)
}