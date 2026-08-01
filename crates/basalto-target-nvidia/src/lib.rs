pub mod codegen;
pub mod runtime;

// Re-exporta a função textual sempre disponível
pub use codegen::generate_ptx;

// Se a feature estiver ativada, re-exporta o módulo LLVM
#[cfg(feature = "llvm-codegen")]
pub use codegen::llvm;