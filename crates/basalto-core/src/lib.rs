// EXTRAÍDO DO FLAGTREE: include/triton/ir/, lib/Analysis/, lib/Transforms/, lib/Dialect/
pub mod ir;
pub mod analysis;
pub mod transforms;
pub mod dialect;
pub mod flir_builder;
pub mod hasher;

pub use flir_builder::build_flir;
pub use hasher::hash_kernel;
