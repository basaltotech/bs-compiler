use std::env;
use std::fs;
use std::path::Path;
use anyhow::Result;
use serde::Deserialize;

// Adicionamos a macro Deserialize para ler o arquivo TOML automaticamente
#[derive(Deserialize, Debug)]
pub struct Config {
    pub redis_cache_url: String,
    pub log_level: String,
}

// Caminho padrão global de sistemas Unix/Linux para aplicativos root
const DEFAULT_CONFIG_PATH: &str = "/etc/basalto/config.toml";

pub fn load_config() -> Result<Config> {
    // 1. Tenta carregar do arquivo de configuração do sistema protegido por root (/etc)
    if Path::new(DEFAULT_CONFIG_PATH).exists() {
        // Como o processo roda como root, ele tem garantia de leitura sem restrições
        let config_str = fs::read_to_string(DEFAULT_CONFIG_PATH)?;
        if let Ok(mut config) = toml::from_str::<Config>(&config_str) {
            // Aplica sobreposição (override) se o usuário definiu variáveis de ambiente específicas no Job
            if let Ok(env_redis) = env::var("REDIS_CACHE_URL") {
                config.redis_cache_url = env_redis;
            }
            if let Ok(env_log) = env::var("LOG_LEVEL") {
                config.log_level = env_log;
            }
            return Ok(config);
        }
    }

    // 2. Fallback: Se o arquivo no /etc não existir, lê estritamente do ambiente (comportamento original)
    Ok(Config {
        redis_cache_url: env::var("REDIS_CACHE_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string()),
        log_level: env::var("LOG_LEVEL")
            .unwrap_or_else(|_| "info".to_string()),
    })
}
