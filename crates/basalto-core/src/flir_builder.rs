use anyhow::{anyhow, Result};
use inkwell::{
    AddressSpace, IntPredicate,
    context::Context,
    targets::{Target, TargetMachine, TargetTriple, InitializationConfig, FileType},
    memory_buffer::MemoryBuffer,
};
use basalto_common::hardware::DeviceCapabilities;
use serde_json::Value;

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

/// Constrói o módulo FLIR. `dtype` pode ser "f32" ou "f64".
pub fn build_flir(_graph_str: &str, caps: &Option<DeviceCapabilities>, dtype: &str) -> Result<FlirModule> {
    let tile_size = caps.as_ref().map(|c| c.max_threads_per_block as i64).unwrap_or(128);
    let radius = 1;
    let elem_size = if dtype == "f32" { 4 } else { 8 };
    let shared_mem_bytes = ((tile_size + 2 * radius) as u32) * elem_size;

    let ops = vec![FlirOp {
        op: "stencil_1d".to_string(),
        inputs: vec!["x".to_string()],
        output: "y".to_string(),
        params: Some(serde_json::json!({
            "radius": radius,
            "coeffs": [0.2, 0.3, 0.5],
            "tile_size": tile_size,
            "shared_mem_bytes": shared_mem_bytes,
            "dtype": dtype,
        })),
    }];
    Ok(FlirModule { ops, input_names: vec!["x".to_string()], output_names: vec!["y".to_string()] })
}

pub fn flir_to_llvm(module: &FlirModule, caps: &Option<DeviceCapabilities>, dtype: &str) -> Result<String> {
    let op = module.ops.first().ok_or_else(|| anyhow!("Nenhuma operação"))?;
    let params = op.params.as_ref().ok_or_else(|| anyhow!("Sem params"))?;

    let radius: i64 = params["radius"].as_i64().unwrap_or(1);
    let coeffs: Vec<f64> = params["coeffs"].as_array()
        .unwrap_or(&vec![serde_json::Value::from(1.0)])
        .iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
    let tile_size: i64 = params["tile_size"].as_i64().unwrap_or(128);
    let _shared_mem_bytes = params["shared_mem_bytes"].as_u64().unwrap_or(0) as u32;

    // Inicializa NVPTX
    Target::initialize_nvptx(&InitializationConfig::default());
    let context = Context::create();
    let llvm_module = context.create_module("basalto_kernel");
    llvm_module.set_triple(&TargetTriple::create("nvptx64-nvidia-cuda"));

    let f64_type = context.f64_type();
    let f32_type = context.f32_type();
    let float_type = if dtype == "f32" { f32_type } else { f64_type };
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    let void_type = context.void_type();

    let generic_ptr = float_type.ptr_type(AddressSpace(0));
    let fn_type = void_type.fn_type(&[generic_ptr.into(), generic_ptr.into(), i32_type.into()], false);
    let kernel_fn = llvm_module.add_function("basalto_kernel", fn_type, None);
    let x_ptr = kernel_fn.get_param(0).unwrap().into_pointer_value();
    let y_ptr = kernel_fn.get_param(1).unwrap().into_pointer_value();
    let n_param = kernel_fn.get_param(2).unwrap().into_int_value();

    let entry = context.append_basic_block(kernel_fn, "entry");
    let builder = context.create_builder();
    builder.position_at_end(entry);

    let x_global = builder.build_address_space_cast(x_ptr, float_type.ptr_type(AddressSpace(1)), "x_global");
    let y_global = builder.build_address_space_cast(y_ptr, float_type.ptr_type(AddressSpace(1)), "y_global");

    // Shared memory
    let shared_type = float_type.array_type(0);
    let shared_global = llvm_module.add_global(shared_type, Some(AddressSpace(3)), "shared_mem");
    shared_global.set_linkage(inkwell::module::Linkage::External);
    shared_global.set_alignment(if dtype == "f32" { 4 } else { 8 });
    let base_ptr = shared_global.as_pointer_value();

    let i32_fn_type = i32_type.fn_type(&[], false);
    let tid_fn = llvm_module.add_function("llvm.nvvm.read.ptx.sreg.tid.x", i32_fn_type, None);
    let bid_fn = llvm_module.add_function("llvm.nvvm.read.ptx.sreg.ctaid.x", i32_fn_type, None);
    let bdim_fn = llvm_module.add_function("llvm.nvvm.read.ptx.sreg.ntid.x", i32_fn_type, None);
    let barrier_fn = llvm_module.add_function("llvm.nvvm.barrier0", void_type.fn_type(&[], false), None);

    let tid = builder.build_call(tid_fn, &[], "tid").try_as_basic_value().left().unwrap().into_int_value();
    let bid = builder.build_call(bid_fn, &[], "bid").try_as_basic_value().left().unwrap().into_int_value();
    let bdim = builder.build_call(bdim_fn, &[], "bdim").try_as_basic_value().left().unwrap().into_int_value();

    let tid64 = builder.build_int_cast(tid, i64_type, "tid64");
    let bid64 = builder.build_int_cast(bid, i64_type, "bid64");
    let bdim64 = builder.build_int_cast(bdim, i64_type, "bdim64");
    let n64 = builder.build_int_cast(n_param, i64_type, "n64");

    let const_i64 = |v: i64| i64_type.const_int(v as u64, false);

    let tile_start = builder.build_int_mul(bid64, bdim64, "tile_start");
    let global_idx = builder.build_int_add(tile_start, tid64, "global_idx");

    let cond_out = builder.build_int_compare(IntPredicate::UGE, global_idx, n64, "cond_out");
    let exit = context.append_basic_block(kernel_fn, "exit");
    let body = context.append_basic_block(kernel_fn, "body");
    builder.build_conditional_branch(cond_out, exit, body);
    builder.position_at_end(body);

    let zero = float_type.const_float(0.0);

    let safe_load = |idx: inkwell::values::IntValue,
                     builder: &inkwell::builder::Builder,
                     x_global: inkwell::values::PointerValue,
                     n64: inkwell::values::IntValue,
                     zero: inkwell::values::FloatValue|
     -> inkwell::values::FloatValue {
        let neg = builder.build_int_compare(IntPredicate::SLT, idx, const_i64(0), "neg");
        let ge_n = builder.build_int_compare(IntPredicate::UGE, idx, n64, "ge_n");
        let invalid = builder.build_or(neg, ge_n, "invalid");
        let ptr = unsafe { builder.build_gep(x_global, &[idx], "ptr") };
        let loaded = builder.build_load(ptr, "loaded").into_float_value();
        builder.build_select(invalid, zero, loaded, "safe_val").into_float_value()
    };

    // Center
    let center_idx = builder.build_int_add(tid64, const_i64(radius), "center_idx");
    let center_val = safe_load(global_idx, &builder, x_global, n64, zero);
    let center_store = unsafe { builder.build_gep(base_ptr, &[const_i64(0), center_idx], "center_store") };
    builder.build_store(center_store, center_val);

    // Halo esquerdo
    let left_cond = builder.build_int_compare(IntPredicate::SLT, tid64, const_i64(radius), "left_cond");
    let left_block = context.append_basic_block(kernel_fn, "left_halo");
    let after_left = context.append_basic_block(kernel_fn, "after_left");
    builder.build_conditional_branch(left_cond, left_block, after_left);
    builder.position_at_end(left_block);
    {
        let left_global_idx = builder.build_int_sub(builder.build_int_add(tile_start, tid64, "tmp"), const_i64(radius), "left_global_idx");
        let left_val = safe_load(left_global_idx, &builder, x_global, n64, zero);
        let left_store = unsafe { builder.build_gep(base_ptr, &[const_i64(0), tid64], "left_store") };
        builder.build_store(left_store, left_val);
    }
    builder.build_unconditional_branch(after_left);
    builder.position_at_end(after_left);

    // Halo direito
    let right_threshold = builder.build_int_sub(bdim64, const_i64(radius), "right_threshold");
    let right_cond = builder.build_int_compare(IntPredicate::UGE, tid64, right_threshold, "right_cond");
    let right_block = context.append_basic_block(kernel_fn, "right_halo");
    let after_right = context.append_basic_block(kernel_fn, "after_right");
    builder.build_conditional_branch(right_cond, right_block, after_right);
    builder.position_at_end(right_block);
    {
        let right_offset = builder.build_int_sub(tid64, right_threshold, "right_offset");
        let right_global_idx = builder.build_int_add(builder.build_int_add(tile_start, bdim64, "tile_plus_bdim"), right_offset, "right_global_idx");
        let right_val = safe_load(right_global_idx, &builder, x_global, n64, zero);
        let right_shared_idx = builder.build_int_add(builder.build_int_add(const_i64(radius), bdim64, "radius_plus_bdim"), right_offset, "right_shared_idx");
        let right_store = unsafe { builder.build_gep(base_ptr, &[const_i64(0), right_shared_idx], "right_store") };
        builder.build_store(right_store, right_val);
    }
    builder.build_unconditional_branch(after_right);
    builder.position_at_end(after_right);

    builder.build_call(barrier_fn, &[], "sync_after_load");

    let total_shared_elems = builder.build_int_add(bdim64, const_i64(2 * radius), "total_shared_elems");
    let mut result = float_type.const_float(0.0);
    for (r, coeff) in coeffs.iter().enumerate() {
        let r_offset = (r as i64) - radius;
        let coeff_val = float_type.const_float(*coeff);
        let neighbor_idx = builder.build_int_add(center_idx, const_i64(r_offset), "neighbor_idx");
        let valid_low = builder.build_int_compare(IntPredicate::SGE, neighbor_idx, const_i64(0), "valid_low");
        let valid_high = builder.build_int_compare(IntPredicate::SLT, neighbor_idx, total_shared_elems, "valid_high");
        let valid = builder.build_and(valid_low, valid_high, "valid");
        let safe_idx = builder.build_select(valid, neighbor_idx, const_i64(0), "safe_idx").into_int_value();
        let ptr = unsafe { builder.build_gep(base_ptr, &[const_i64(0), safe_idx], "ptr") };
        let val = builder.build_load(ptr, "val").into_float_value();
        let weighted = builder.build_float_mul(val, coeff_val, "weighted");
        result = builder.build_float_add(result, weighted, "accum");
    }
    builder.build_call(barrier_fn, &[], "sync_after_compute");
    let out_ptr = unsafe { builder.build_gep(y_global, &[global_idx], "out_ptr") };
    builder.build_store(out_ptr, result);
    builder.build_unconditional_branch(exit);
    builder.position_at_end(exit);
    builder.build_return(None);

    let func_meta = kernel_fn.as_metadata_value();
    let kernel_str = context.metadata_string("kernel");
    let one_i32 = context.i32_type().const_int(1, false).as_metadata_value();
    let md_node = context.metadata_node(&[func_meta.into(), kernel_str.into(), one_i32.into()]);
    llvm_module.add_named_metadata("nvvm.annotations", &[md_node]);

    Ok(llvm_module.print_to_string().to_string())
}

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