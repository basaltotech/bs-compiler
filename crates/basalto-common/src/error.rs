use thiserror::Error;

#[derive(Debug, Error)]
pub enum BasaltoError {
    #[error("Falha ao detectar hardware: {0}")]
    Hardware(String),
    #[error("Erro de permissão: {0}")]
    Permission(String),
    #[error("Erro de configuração: {0}")]
    Config(String),
    #[error("Erro de telemetria: {0}")]
    Telemetry(String),
}