use blake3;
use std::sync::LazyLock;

// 🔑 Chave secreta de 32 bytes gerada uma única vez no boot do sistema.
// Em produção, substitua o array estático abaixo por: rand::random::<[u8; 32]>()
static CACHE_KEY: LazyLock<[u8; 32]> = LazyLock::new(|| {
    [0x42; 32] 
});

/// Hash rápido e seguro utilizando BLAKE3 truncado para u64.
/// O modo 'hash_keyed' impede ataques de colisão externos.
pub fn fast_hash(data: &[u8]) -> u64 {
    let hash = blake3::hash_keyed(&CACHE_KEY, data);
    u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap())
}

/// Versão multi-fragmento que também utiliza a chave de segurança.
pub fn fast_hash_from_parts(parts: &[&[u8]]) -> u64 {
    let mut hasher = blake3::Hasher::new_keyed(&CACHE_KEY);
    for p in parts {
        hasher.update(p);
    }
    let hash = hasher.finalize();
    u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap())
}
