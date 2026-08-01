use blake3;
use std::sync::LazyLock;

static CACHE_KEY: LazyLock<[u8; 32]> = LazyLock::new(|| {
    [0x42; 32] // chave secreta fixa para demonstração
});

pub fn fast_hash(data: &[u8]) -> u64 {
    let hash = blake3::hash_keyed(&CACHE_KEY, data);
    u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap())
}

pub fn fast_hash_from_parts(parts: &[&[u8]]) -> u64 {
    let mut hasher = blake3::Hasher::new_keyed(&CACHE_KEY);
    for p in parts {
        hasher.update(p);
    }
    let hash = hasher.finalize();
    u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap())
}

// ⭐ NOVA FUNÇÃO exigida pelo interceptor
pub fn hash_kernel(graph: &str, shapes: &[Vec<usize>], vendor: &str, arch: &str, driver: &str) -> u64 {
    let mut hasher = blake3::Hasher::new_keyed(&CACHE_KEY);
    
    hasher.update(graph.as_bytes());
    
    // Serializa os shapes em little-endian
    for shape in shapes {
        for dim in shape {
            hasher.update(&dim.to_le_bytes());
        }
    }
    
    hasher.update(vendor.as_bytes());
    hasher.update(arch.as_bytes());
    hasher.update(driver.as_bytes());
    
    let hash = hasher.finalize();
    u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap())
}