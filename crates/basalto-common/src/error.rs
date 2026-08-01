use thiserror::Error;
use pyo3::prelude::*;
use pyo3::exceptions::{PyPermissionError, PyRuntimeError, PyValueError};

#[derive(Error, Debug)]
pub enum BasaltoError {
    #[error("Privilégios de root necessários: {0}")]
    RootRequired(String),

    #[error("Falha de hardware/driver na GPU: {0}")]
    Hardware(String),

    #[error("Erro de cache (Local/Cluster): {0}")]
    Cache(String),

    #[error("Erro de compilação do Kernel: {0}")]
    Compilation(String),
}

// 🐍 PONTE PYTHON (PyO3): Traduz os erros do Rust para Exceções nativas do Python automaticamente!
// Com este bloco, você pode usar o operador `?` diretamente em funções retornando `PyResult`.
impl From<BasaltoError> for PyErr {
    fn from(error: BasaltoError) -> Self {
        match error {
            // Se faltar root, joga um PermissionError no Python
            BasaltoError::RootRequired(msg) => PyPermissionError::new_err(msg),
            
            // Parâmetros inválidos de hardware viram ValueError
            BasaltoError::Hardware(msg) if msg.contains("inválido") => PyValueError::new_err(msg),
            
            // Falhas críticas de hardware, cache ou JIT viram RuntimeError
            BasaltoError::Hardware(msg) => PyRuntimeError::new_err(msg),
            BasaltoError::Cache(msg) => PyRuntimeError::new_err(msg),
            BasaltoError::Compilation(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

// 🛠️ CONVERSÕES AUTOMÁTICAS: Permite que erros externos virem variantes do Basalto via operador `?`
impl From<std::io::Error> for BasaltoError {
    fn from(err: std::io::Error) -> Self {
        BasaltoError::Cache(format!("Falha de I/O no disco: {}", err))
    }
}

// Exemplo caso use a biblioteca Redis oficial do cluster
#[cfg(feature = "redis")]
impl From<redis::RedisError> for BasaltoError {
    fn from(err: redis::RedisError) -> Self {
        BasaltoError::Cache(format!("Falha de comunicação com o Redis: {}", err))
    }
}
