use anyhow::{anyhow, Result};
use basalto_common::hardware::DeviceCapabilities;
use serde_json::Value;
use inkwell::context::Context;
use inkwell::targets::{Target, TargetTriple, InitializationConfig, FileType};
use inkwell::memory_buffer::MemoryBuffer;
use crate::ir::{get_generator, tensor_core::Stencil3DTensorCore, StencilGenerator};

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

pub fn build_flir(
    _graph_str: &str,
    caps: &Option<DeviceCapabilities>,
    dtype: &str,
    shape: &[usize],
    strides: &[isize],
    radius: usize,
    custom_coeffs: Option<Vec<f64>>, 
) -> Result<FlirModule> {
    let dims = shape.len();
    let tile_size = caps.as_ref().map(|c| c.max_threads_per_block as i64).unwrap_or(128);
    let elem_size = if dtype == "f32" || dtype == "f16" || dtype == "bf16" { 4 } else { 8 };
    let radius_i64 = radius as i64;

    let (tile_x, tile_y) = if dims >= 2 {
        let t = (tile_size as f64).sqrt() as i64;
        (t, t)
    } else {
        (tile_size, 1)
    };

    let shared_mem_bytes = if dims == 1 {
        ((tile_size + 2 * radius_i64) as u32) * elem_size
    } else {
        ((tile_x + 2 * radius_i64) * (tile_y + 2 * radius_i64)) as u32 * elem_size
    };

    let op_name = match dims {
        1 => format!("stencil_1d_r{}", radius),
        2 => format!("stencil_2d_r{}", radius),
        3 => format!("stencil_3d_r{}", radius),
        _ => return Err(anyhow!("Dimensão {} não suportada", dims)),
    };

    // Coeficientes: se fornecidos pelo Python, usa-os; senão, gera isotrópicos.
    let coeffs = if let Some(coeffs) = custom_coeffs {
        coeffs
    } else {
        let total_coeffs = (2 * radius + 1).pow(dims as u32);
        let c = 1.0 / total_coeffs as f64;
        vec![c; total_coeffs as usize]
    };

    let stride_x = if dims >= 1 { strides[0] as i64 } else { 1 };
    let stride_y = if dims >= 2 { strides[1] as i64 } else { 1 };
    let stride_z = if dims >= 3 { strides[2] as i64 } else { 1 };

    let ops = vec![FlirOp {
        op: op_name,
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
            "stride_x": stride_x,
            "stride_y": stride_y,
            "stride_z": stride_z,
        })),
    }];

    Ok(FlirModule { ops, input_names: vec!["x".to_string()], output_names: vec!["y".to_string()] })
}

pub fn flir_to_llvm(
    module: &FlirModule,
    caps: &Option<DeviceCapabilities>,
    dtype: &str,
) -> Result<String> {
    let op = module.ops.first().ok_or_else(|| anyhow!("Nenhuma operação"))?;
    let params = op.params.as_ref().ok_or_else(|| anyhow!("Sem params"))?;
    let dims = params["dims"].as_u64().unwrap_or(1) as usize;

    Target::initialize_nvptx(&InitializationConfig::default());
    let context = Context::create();
    let llvm_module = context.create_module("basalto_kernel");
    llvm_module.set_triple(&TargetTriple::create("nvptx64-nvidia-cuda"));

    let use_tensor_core = dims == 3 && (dtype == "f16" || dtype == "bf16");

    let generator: Box<dyn StencilGenerator> = if use_tensor_core {
        eprintln!("[FLIR] Usando Tensor Core para 3D com dtype={}", dtype);
        Box::new(Stencil3DTensorCore)
    } else {
        get_generator(dims)
    };

    generator.generate_ir(&llvm_module, params, caps, dtype, &context)
}

pub fn compile_to_ptx(llvm_ir: &str, caps: &Option<DeviceCapabilities>) -> Result<Vec<u8>> {
    Target::initialize_nvptx(&InitializationConfig::default());
    let target_triple = TargetTriple::create("nvptx64-nvidia-cuda");
    let target = Target::from_triple(&target_triple).map_err(|e| anyhow!("Target error: {:?}", e))?;

    let target_cpu = caps
        .as_ref()
        .map(|c| format!("sm_{}{}", c.compute_capability_major, c.compute_capability_minor))
        .unwrap_or_else(|| "sm_70".to_string());

    let target_machine = target
        .create_target_machine(
            &target_triple,
            &target_cpu,
            "",
            inkwell::targets::CodeGenOptLevel::Aggressive,
            inkwell::targets::RelocMode::Default,
            inkwell::targets::CodeModel::Default,
        )
        .ok_or_else(|| anyhow!("Target machine error"))?;

    let context = Context::create();
    let mem_buffer = MemoryBuffer::create_from_memory_range_copy(llvm_ir.as_bytes(), "basalto_ir");
    let module = context
        .create_module_from_ir(mem_buffer)
        .map_err(|e| anyhow!("Module creation error: {}", e))?;

    let output_buffer = target_machine
        .write_to_memory_buffer(&module, FileType::Assembly)
        .map_err(|e| anyhow!("PTX emission error: {}", e))?;
    Ok(output_buffer.as_slice().to_vec())
}