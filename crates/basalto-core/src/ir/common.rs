//! Utilitários comuns para geração de IR NVPTX.

use inkwell::{
    AddressSpace, IntPredicate,
    context::Context,
    module::Module,
    values::{FloatValue, IntValue, PointerValue, FunctionValue},
    types::{FloatType, IntType, VoidType},
};
use anyhow::Result;

/// Declara a memória compartilhada dinâmica como uma variável global externa.
pub fn declare_shared_memory(
    module: &Module,
    float_type: FloatType,
    dtype: &str,
) -> PointerValue {
    let array_type = float_type.array_type(0);
    let global = module.add_global(array_type, Some(AddressSpace(3)), "shared_mem");
    global.set_linkage(inkwell::module::Linkage::External);
    let align = if dtype == "f32" { 4 } else { 8 };
    global.set_alignment(align);
    global.as_pointer_value()
}

/// Declara os intrínsecos NVPTX para leitura de threadIdx, blockIdx, blockDim e a barreira.
pub fn declare_nvptx_intrinsics(
    module: &Module,
    i32_type: IntType,
    void_type: VoidType,
) -> (FunctionValue, FunctionValue, FunctionValue, FunctionValue, FunctionValue, FunctionValue) {
    let i32_fn_type = i32_type.fn_type(&[], false);
    let void_fn_type = void_type.fn_type(&[], false);

    // para eixo X
    let tid_x = module.add_function("llvm.nvvm.read.ptx.sreg.tid.x", i32_fn_type, None);
    let bid_x = module.add_function("llvm.nvvm.read.ptx.sreg.ctaid.x", i32_fn_type, None);
    let bdim_x = module.add_function("llvm.nvvm.read.ptx.sreg.ntid.x", i32_fn_type, None);
    // para eixo Y
    let tid_y = module.add_function("llvm.nvvm.read.ptx.sreg.tid.y", i32_fn_type, None);
    let bid_y = module.add_function("llvm.nvvm.read.ptx.sreg.ctaid.y", i32_fn_type, None);
    let bdim_y = module.add_function("llvm.nvvm.read.ptx.sreg.ntid.y", i32_fn_type, None);
    let barrier = module.add_function("llvm.nvvm.barrier0", void_fn_type, None);

    (tid_x, bid_x, bdim_x, tid_y, bid_y, bdim_y, barrier)
}

/// Função auxiliar para carregar da memória global com bounds check (zero‑padding).
pub fn safe_load_global<'ctx>(
    builder: &inkwell::builder::Builder<'ctx>,
    x_global: PointerValue<'ctx>,
    idx: IntValue<'ctx>,
    n64: IntValue<'ctx>,
    zero: FloatValue<'ctx>,
    const_i64: impl Fn(i64) -> IntValue<'ctx>,
) -> FloatValue<'ctx> {
    let neg = builder.build_int_compare(IntPredicate::SLT, idx, const_i64(0), "neg");
    let ge_n = builder.build_int_compare(IntPredicate::UGE, idx, n64, "ge_n");
    let invalid = builder.build_or(neg, ge_n, "invalid");
    let ptr = unsafe { builder.build_gep(x_global, &[idx], "ptr") };
    let loaded = builder.build_load(ptr, "loaded").into_float_value();
    builder.build_select(invalid, zero, loaded, "safe_val").into_float_value()
}