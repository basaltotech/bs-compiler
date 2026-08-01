use inkwell::types::{BasicTypeEnum, FloatType, IntType, StructType, VoidType};
use inkwell::context::Context;

pub fn get_float_type(ctx: &Context) -> FloatType {
    ctx.f32_type() // ou f64, dependendo do tensor
}

pub fn get_int_type(ctx: &Context, bits: u32) -> IntType {
    ctx.i_type(bits)
}

pub fn get_void_type(ctx: &Context) -> VoidType {
    ctx.void_type()
}