//! Orquestrador do FLIR: usa a factory para escolher o gerador adequado.

use anyhow::{anyhow, Result};
use basalto_common::hardware::DeviceCapabilities;
use serde_json::Value;
use inkwell::context::Context;
use inkwell::targets::{Target, TargetTriple, InitializationConfig, FileType};
use inkwell::memory_buffer::MemoryBuffer;
use crate::ir::{get_generator, StencilGenerator};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FlirOp {
    pub op: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub params: Option<Value>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FlirModule {
    pub ops: Vec<FlirOp>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

/// Constrói o módulo FLIR a partir do grafo e shape.
pub fn build_flir(
    _graph_str: &str,
    caps: &Option<DeviceCapabilities>,
    dtype: &str,
    shape: &[usize],
) -> Result<FlirModule> {
    let dims = shape.len();
    let tile_size = caps.as_ref().map(|c| c.max_threads_per_block as i64).unwrap_or(128);
    let radius = 1;
    let elem_size = if dtype == "f32" { 4 } else { 8 };

    // Para 2D/3D, usamos tile_x e tile_y (sqrt do total)
    let (tile_x, tile_y) = if dims >= 2 {
        let t = (tile_size as f64).sqrt() as i64;
        (t, t)
    } else {
        (tile_size, 1)
    };

    // Memória compartilhada para a tile 2D (X-Y) com halo
    // Para 3D, a shared memory guarda apenas a fatia X-Y (Z é loop)
    let shared_mem_bytes = if dims == 1 {
        ((tile_size + 2 * radius) as u32) * elem_size
    } else {
        ((tile_x + 2 * radius) * (tile_y + 2 * radius)) as u32 * elem_size
    };

    let op_name = match dims {
        1 => "stencil_1d",
        2 => "stencil_2d",
        3 => "stencil_3d",
        _ => return Err(anyhow!("Dimensão {} não suportada", dims)),
    };

    // Coeficientes
    let coeffs = if dims == 1 {
        vec![0.2, 0.3, 0.5] // 1D: 3 coeffs
    } else if dims == 2 {
        // 2D: 3x3 = 9 coeffs
        vec![
            0.1, 0.2, 0.1,
            0.2, 0.0, 0.2,
            0.1, 0.2, 0.1,
        ]
    } else {
        // 3D: 3x3x3 = 27 coeffs (isotrópico simples)
        let c = 1.0 / 27.0;
        vec![c; 27]
    };

    let ops = vec![FlirOp {
        op: op_name.to_string(),
        inputs: vec!["x".to_string()],
        output: "y".to_string(),
        params: Some(serde_json::json!({
            "radius": radius,
            "coeffs": coeffs,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "shared_mem_bytes": shared_mem_bytes,
            "dtype": dtype,
            "dims": dims,
        })),
    }];

    Ok(FlirModule {
        ops,
        input_names: vec!["x".to_string()],
        output_names: vec!["y".to_string()],
    })
}

/// Gera o LLVM IR utilizando o gerador adequado.
pub fn flir_to_llvm(module: &FlirModule, caps: &Option<DeviceCapabilities>, dtype: &str) -> Result<String> {
    let op = module.ops.first().ok_or_else(|| anyhow!("Nenhuma operação"))?;
    let params = op.params.as_ref().ok_or_else(|| anyhow!("Sem params"))?;
    let dims = params["dims"].as_u64().unwrap_or(1) as usize;

    Target::initialize_nvptx(&InitializationConfig::default());
    let context = Context::create();
    let llvm_module = context.create_module("basalto_kernel");
    llvm_module.set_triple(&TargetTriple::create("nvptx64-nvidia-cuda"));

    let generator = get_generator(dims);
    generator.generate_ir(&llvm_module, params, caps, dtype, &context)
}

/// Compila LLVM IR → PTX.
pub fn compile_to_ptx(llvm_ir: &str, caps: &Option<DeviceCapabilities>) -> Result<Vec<u8>> {
    Target::initialize_nvptx(&InitializationConfig::default());
    let target_triple = TargetTriple::create("nvptx64-nvidia-cuda");
    let target = Target::from_triple(&target_triple).map_err(|e| anyhow!("Target error: {:?}", e))?;

    let target_cpu = caps.as_ref()
        .map(|c| format!("sm_{}{}", c.compute_capability_major, c.compute_capability_minor))
        .unwrap_or_else(|| "sm_70".to_string());

    let target_machine = target.create_target_machine(
        &target_triple, &target_cpu, "",
        inkwell::targets::CodeGenOptLevel::Aggressive,
        inkwell::targets::RelocMode::Default,
        inkwell::targets::CodeModel::Default,
    ).ok_or_else(|| anyhow!("Target machine error"))?;

    let context = Context::create();
    let mem_buffer = MemoryBuffer::create_from_memory_range_copy(llvm_ir.as_bytes(), "basalto_ir");
    let module = context.create_module_from_ir(mem_buffer)
        .map_err(|e| anyhow!("Module creation error: {}", e))?;

    let output_buffer = target_machine.write_to_memory_buffer(&module, FileType::Assembly)
        .map_err(|e| anyhow!("PTX emission error: {}", e))?;
    Ok(output_buffer.as_slice().to_vec())
}