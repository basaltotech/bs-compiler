//! Implementação do stencil 1D com memória compartilhada.

use super::common;
use super::StencilGenerator;
use inkwell::{
    AddressSpace, IntPredicate,
    context::Context,
    module::Module,
    values::FloatValue,
};
use anyhow::{anyhow, Result};
use serde_json::Value;
use basalto_common::hardware::DeviceCapabilities;

pub struct Stencil1D;

impl StencilGenerator for Stencil1D {
    fn generate_ir(
        &self,
        module: &Module,
        params: &Value,
        _caps: &Option<DeviceCapabilities>,
        dtype: &str,
        ctx: &Context,
    ) -> Result<String> {
        let radius = params["radius"].as_i64().unwrap_or(1);
        let coeffs: Vec<f64> = params["coeffs"].as_array()
            .unwrap_or(&vec![serde_json::Value::from(1.0)])
            .iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();

        // Tipos
        let f64 = ctx.f64_type();
        let f32 = ctx.f32_type();
        let float = if dtype == "f32" { f32 } else { f64 };
        let i32 = ctx.i32_type();
        let i64 = ctx.i64_type();
        let void = ctx.void_type();

        // Assinatura: kernel(double* x, double* y, int N)
        let generic_ptr = float.ptr_type(AddressSpace(0));
        let fn_type = void.fn_type(&[generic_ptr.into(), generic_ptr.into(), i32.into()], false);
        let kernel = module.add_function("basalto_kernel", fn_type, None);
        let x_ptr = kernel.get_param(0).unwrap().into_pointer_value();
        let y_ptr = kernel.get_param(1).unwrap().into_pointer_value();
        let n_param = kernel.get_param(2).unwrap().into_int_value();

        let entry = ctx.append_basic_block(kernel, "entry");
        let builder = ctx.create_builder();
        builder.position_at_end(entry);

        let x_global = builder.build_address_space_cast(x_ptr, float.ptr_type(AddressSpace(1)), "x_global");
        let y_global = builder.build_address_space_cast(y_ptr, float.ptr_type(AddressSpace(1)), "y_global");

        // Shared memory e intrínsecos
        let shared = common::declare_shared_memory(module, float, dtype);
        let (tid, bid, bdim, _, _, _, barrier) = common::declare_nvptx_intrinsics(module, i32, void);

        // Ler registros
        let tid_val = builder.build_call(tid, &[], "tid").try_as_basic_value().left().unwrap().into_int_value();
        let bid_val = builder.build_call(bid, &[], "bid").try_as_basic_value().left().unwrap().into_int_value();
        let bdim_val = builder.build_call(bdim, &[], "bdim").try_as_basic_value().left().unwrap().into_int_value();

        let tid64 = builder.build_int_cast(tid_val, i64, "tid64");
        let bid64 = builder.build_int_cast(bid_val, i64, "bid64");
        let bdim64 = builder.build_int_cast(bdim_val, i64, "bdim64");
        let n64 = builder.build_int_cast(n_param, i64, "n64");

        let const_i64 = |v: i64| i64.const_int(v as u64, false);

        let tile_start = builder.build_int_mul(bid64, bdim64, "tile_start");
        let global_idx = builder.build_int_add(tile_start, tid64, "global_idx");

        // Bounds check
        let cond_out = builder.build_int_compare(IntPredicate::UGE, global_idx, n64, "cond_out");
        let exit_block = ctx.append_basic_block(kernel, "exit");
        let body_block = ctx.append_basic_block(kernel, "body");
        builder.build_conditional_branch(cond_out, exit_block, body_block);
        builder.position_at_end(body_block);

        let zero = float.const_float(0.0);

        // safe_load helper local
        let safe_load = |idx: inkwell::values::IntValue| -> FloatValue {
            common::safe_load_global(&builder, x_global, idx, n64, zero, const_i64)
        };

        // Carregar central
        let center_idx = builder.build_int_add(tid64, const_i64(radius), "center_idx");
        let center_val = safe_load(global_idx);
        let store_center = unsafe { builder.build_gep(shared, &[const_i64(0), center_idx], "store_center") };
        builder.build_store(store_center, center_val);

        // Halo esquerdo
        let left_cond = builder.build_int_compare(IntPredicate::SLT, tid64, const_i64(radius), "left_cond");
        let left_block = ctx.append_basic_block(kernel, "left_halo");
        let after_left = ctx.append_basic_block(kernel, "after_left");
        builder.build_conditional_branch(left_cond, left_block, after_left);
        builder.position_at_end(left_block);
        {
            let left_global = builder.build_int_sub(builder.build_int_add(tile_start, tid64, "tmp"), const_i64(radius), "left_global");
            let left_val = safe_load(left_global);
            let left_store = unsafe { builder.build_gep(shared, &[const_i64(0), tid64], "left_store") };
            builder.build_store(left_store, left_val);
        }
        builder.build_unconditional_branch(after_left);
        builder.position_at_end(after_left);

        // Halo direito
        let right_threshold = builder.build_int_sub(bdim64, const_i64(radius), "right_threshold");
        let right_cond = builder.build_int_compare(IntPredicate::UGE, tid64, right_threshold, "right_cond");
        let right_block = ctx.append_basic_block(kernel, "right_halo");
        let after_right = ctx.append_basic_block(kernel, "after_right");
        builder.build_conditional_branch(right_cond, right_block, after_right);
        builder.position_at_end(right_block);
        {
            let right_offset = builder.build_int_sub(tid64, right_threshold, "right_offset");
            let right_global = builder.build_int_add(builder.build_int_add(tile_start, bdim64, "tile_plus_bdim"), right_offset, "right_global");
            let right_val = safe_load(right_global);
            let right_shared_idx = builder.build_int_add(
                builder.build_int_add(const_i64(radius), bdim64, "radius_plus_bdim"),
                right_offset,
                "right_shared_idx"
            );
            let right_store = unsafe { builder.build_gep(shared, &[const_i64(0), right_shared_idx], "right_store") };
            builder.build_store(right_store, right_val);
        }
        builder.build_unconditional_branch(after_right);
        builder.position_at_end(after_right);

        // Sincronizar
        builder.build_call(barrier, &[], "sync_after_load");

        // Stencil
        let total_elems = builder.build_int_add(bdim64, const_i64(2 * radius), "total_elems");
        let mut result = float.const_float(0.0);
        for (r, coeff) in coeffs.iter().enumerate() {
            let r_off = (r as i64) - radius;
            let coeff_val = float.const_float(*coeff);
            let neighbor_idx = builder.build_int_add(center_idx, const_i64(r_off), "neighbor_idx");
            let valid_low = builder.build_int_compare(IntPredicate::SGE, neighbor_idx, const_i64(0), "valid_low");
            let valid_high = builder.build_int_compare(IntPredicate::SLT, neighbor_idx, total_elems, "valid_high");
            let valid = builder.build_and(valid_low, valid_high, "valid");
            let safe_idx = builder.build_select(valid, neighbor_idx, const_i64(0), "safe_idx").into_int_value();
            let ptr = unsafe { builder.build_gep(shared, &[const_i64(0), safe_idx], "neighbor_ptr") };
            let val = builder.build_load(ptr, "neighbor_val").into_float_value();
            let weighted = builder.build_float_mul(val, coeff_val, "weighted");
            result = builder.build_float_add(result, weighted, "accum");
        }
        builder.build_call(barrier, &[], "sync_after_compute");

        // Armazenar resultado
        let out_ptr = unsafe { builder.build_gep(y_global, &[global_idx], "out_ptr") };
        builder.build_store(out_ptr, result);
        builder.build_unconditional_branch(exit_block);
        builder.position_at_end(exit_block);
        builder.build_return(None);

        // Metadado NVPTX
        let func_meta = kernel.as_metadata_value();
        let kernel_str = ctx.metadata_string("kernel");
        let one_i32 = ctx.i32_type().const_int(1, false).as_metadata_value();
        let md_node = ctx.metadata_node(&[func_meta.into(), kernel_str.into(), one_i32.into()]);
        module.add_named_metadata("nvvm.annotations", &[md_node]);

        Ok(module.print_to_string().to_string())
    }
}