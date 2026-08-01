use std::env;
use std::fs;
use std::path::PathBuf;

pub fn load_env() {
    dotenvy::dotenv().ok();
}

pub fn get_env_or_default(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

pub fn cache_dir() -> PathBuf {
    dirs::cache_dir().unwrap_or_else(|| PathBuf::from("/tmp")).join("basalto")
}