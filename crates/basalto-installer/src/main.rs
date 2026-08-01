use basalto_common::permissions::ensure_root_or_die;
use basalto_common::hardware::detect_gpu;
use std::path::Path;

fn main() {
    // 1. Exige root (obrigatório)
    if let Err(e) = ensure_root_or_die() {
        eprintln!("{}", e);
        std::process::exit(1);
    }

    // 2. Detecta hardware (GPU)
    let gpu_info = match detect_gpu() {
        Ok(info) => info,
        Err(e) => {
            eprintln!("Falha na detecção de GPU: {}", e);
            std::process::exit(1);
        }
    };
    println!("GPU detectada: {} (arch: {}, driver: {})", 
        gpu_info.vendor, gpu_info.arch, gpu_info.driver_version);

    // 3. (OPCIONAL) Cria diretório de cache local (L1) com permissões adequadas
    let cache_dir = std::env::var("BASALTO_CACHE_DIR")
        .unwrap_or_else(|_| "/var/cache/basalto".to_string());
    if !Path::new(&cache_dir).exists() {
        match std::fs::create_dir_all(&cache_dir) {
            Ok(_) => println!("Diretório de cache criado: {}", cache_dir),
            Err(e) => eprintln!("Aviso: não foi possível criar cache dir: {}", e),
        }
    }

    // 4. (OPCIONAL) Testa conectividade com o Redis (L2), se configurado
    if let Ok(redis_url) = std::env::var("BASALTO_REDIS_URL") {
        match redis::Client::open(redis_url) {
            Ok(client) => {
                match client.get_connection() {
                    Ok(mut conn) => {
                        let _: redis::RedisResult<()> = redis::cmd("PING").query(&mut conn);
                        println!("Redis conectado com sucesso.");
                    }
                    Err(e) => eprintln!("Aviso: Redis não acessível: {}", e),
                }
            }
            Err(e) => eprintln!("Aviso: URL do Redis inválida: {}", e),
        }
    }

    // 5. (OPCIONAL) Define variáveis de ambiente padrão para o runtime
    //    (ex.: BASALTO_VENDOR, BASALTO_ARCH) para que o Python/Rust
    //    não precise redetectá-las a cada execução.
    std::env::set_var("BASALTO_VENDOR", &gpu_info.vendor);
    std::env::set_var("BASALTO_ARCH", &gpu_info.arch);
    // Nota: em HPC, essas variáveis já são exportadas via Slurm, mas é um fallback.

    println!("Instalação concluída com sucesso.");
}