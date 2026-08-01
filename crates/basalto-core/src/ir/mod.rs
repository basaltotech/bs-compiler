//! Módulo de geração de IR para stencils com diferentes dimensionalidades.
//! Segue princípios SOLID: cada dimensão tem sua própria implementação
//! da trait `StencilGenerator`, e a factory decide qual usar.

use inkwell::context::Context;
use inkwell::module::Module;
use anyhow::Result;
use serde_json::Value;
use basalto_common::hardware::DeviceCapabilities;

pub mod common;
pub mod stencil_1d;
pub mod stencil_2d;

/// Trait que todo gerador de stencil deve implementar.
pub trait StencilGenerator {
    /// Gera o LLVM IR para o kernel, escrevendo no módulo fornecido.
    /// Retorna o IR como string (para depuração) ou pode retornar Ok(()) se não precisar.
    fn generate_ir(
        &self,
        module: &Module,
        params: &Value,
        caps: &Option<DeviceCapabilities>,
        dtype: &str,
        ctx: &Context,
    ) -> Result<String>;
}

/// Factory que retorna o gerador apropriado com base no número de dimensões.
pub fn get_generator(dims: usize) -> Box<dyn StencilGenerator> {
    match dims {
        1 => Box::new(stencil_1d::Stencil1D),
        2 => Box::new(stencil_2d::Stencil2D),
        _ => panic!("Dimensão {} ainda não suportada", dims),
    }
}