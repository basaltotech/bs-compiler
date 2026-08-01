use redis::{Client, Commands, SetOptions};
use r2d2::Pool;
use std::sync::LazyLock;
use std::time::Duration;

// 🔗 Pool de Conexões Estático e Global para o Redis do Cluster
// Configurado para reutilizar conexões abertas e evitar o overhead de handshake TCP.
static REDIS_POOL: LazyLock<Option<Pool<Client>>> = LazyLock::new(|| {
    let redis_url = std::env::var("BASALTO_REDIS_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    
    match Client::open(redis_url) {
        Ok(client) => Pool::builder()
            .max_size(16) // Conexões simultâneas por processo
            .connection_timeout(Duration::from_millis(500))
            .build(client)
            .ok(),
        Err(_) => None,
    }
});

/// Retorna o TTL em segundos (padrão 7 dias, ajustável via env)
fn get_ttl() -> usize {
    std::env::var("BASALTO_REDIS_TTL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(604800) // 7 dias
}

pub fn get(hash: u64) -> Option<Vec<u8>> {
    let pool = REDIS_POOL.as_ref()?;
    let mut conn = pool.get().ok()?;
    
    let key = format!("basalto:kernel:{:016x}", hash);
    conn.get(&key).unwrap_or(None)
}

pub fn put(hash: u64, binary: &[u8]) {
    let pool = match REDIS_POOL.as_ref() {
        Some(p) => p.clone(),
        None => return,
    };

    let key = format!("basalto:kernel:{:016x}", hash);
    let binary_vec = binary.to_vec();
    let ttl = get_ttl();

    // ⚡ OTIMIZAÇÃO PARA MPI: Despacha o salvamento para thread separada.
    // Usa SETNX para garantir que apenas o primeiro worker escreva.
    std::thread::spawn(move || {
        let mut conn = match pool.get_timeout(Duration::from_millis(100)) {
            Ok(c) => c,
            Err(_) => return, // Timeout rápido, falha silenciosa
        };

        // ⭐ Comando atômico: só escreve se a chave NÃO existir.
        let _: redis::RedisResult<()> = conn.set_options(
            &key,
            binary_vec,
            SetOptions::default()
                .with_nx()              // SET NX (apenas se não existir)
                .with_ex(ttl as u64),   // Expira após TTL segundos
        );

        // Log opcional (desativado por padrão em HPC para evitar flood)
        if std::env::var("BASALTO_DEBUG_CACHE").is_ok() {
            let job_id = std::env::var("SLURM_JOB_ID").unwrap_or_else(|_| "unknown".to_string());
            eprintln!("[Basalto] Cache escrito (ou ignorado) para job {}", job_id);
        }
    });
}