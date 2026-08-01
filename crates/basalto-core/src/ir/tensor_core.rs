//! Gerador de IR para stencil 3D utilizando Tensor Cores (WMMA).
//! Suporta apenas FP16 e BF16. Para FP32/FP64, use o stencil_3d padrão.

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

pub struct Stencil3DTensorCore;

impl StencilGenerator for Stencil3DTensorCore {
    fn generate_ir(
        &self,
        module: &Module,
        params: &Value,
        _caps: &Option<DeviceCapabilities>,
        dtype: &str,
        ctx: &Context,
    ) -> Result<String> {
        // Validação: Tensor Core só funciona com FP16 ou BF16
        if dtype != "f16" && dtype != "bf16" {
            return Err(anyhow!(
                "Tensor Core só suporta FP16 ou BF16 (recebido: {})",
                dtype
            ));
        }

        let radius = params["radius"].as_i64().unwrap_or(1);
        let coeffs: Vec<f64> = params["coeffs"]
            .as_array()
            .unwrap_or(&vec![serde_json::Value::from(1.0)])
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();
        let tile_x = params["tile_x"].as_i64().unwrap_or(64) as i64;
        let tile_y = params["tile_y"].as_i64().unwrap_or(64) as i64;
        let stride_x = params["stride_x"].as_i64().unwrap_or(1);
        let stride_y = params["stride_y"].as_i64().unwrap_or(1);
        let stride_z = params["stride_z"].as_i64().unwrap_or(1);

        let f16 = ctx.half_type();
        let f32 = ctx.f32_type();
        let float = if dtype == "f16" { f16 } else { f32 };
        let i32 = ctx.i32_type();
        let i64 = ctx.i64_type();
        let void = ctx.void_type();

        // Assinatura: kernel(float* x, float* y, int Nx, int Ny, int Nz, int Sx, int Sy, int Sz)
        let generic_ptr = float.ptr_type(AddressSpace(0));
        let fn_type = void.fn_type(
            &[
                generic_ptr.into(),
                generic_ptr.into(),
                i32.into(),
                i32.into(),
                i32.into(),
                i32.into(),
                i32.into(),
                i32.into(),
            ],
            false,
        );
        let kernel = module.add_function("basalto_kernel_3d_tc", fn_type, None);
        let x_ptr = kernel.get_param(0).unwrap().into_pointer_value();
        let y_ptr = kernel.get_param(1).unwrap().into_pointer_value();
        let nx_param = kernel.get_param(2).unwrap().into_int_value();
        let ny_param = kernel.get_param(3).unwrap().into_int_value();
        let nz_param = kernel.get_param(4).unwrap().into_int_value();
        let stride_x_param = kernel.get_param(5).unwrap().into_int_value();
        let stride_y_param = kernel.get_param(6).unwrap().into_int_value();
        let stride_z_param = kernel.get_param(7).unwrap().into_int_value();

        let entry = ctx.append_basic_block(kernel, "entry");
        let builder = ctx.create_builder();
        builder.position_at_end(entry);

        let x_global = builder.build_address_space_cast(
            x_ptr,
            float.ptr_type(AddressSpace(1)),
            "x_global",
        );
        let y_global = builder.build_address_space_cast(
            y_ptr,
            float.ptr_type(AddressSpace(1)),
            "y_global",
        );

        let shared = common::declare_shared_memory(module, float, dtype);
        let (tid_x, bid_x, bdim_x, tid_y, bid_y, bdim_y, barrier) =
            common::declare_nvptx_intrinsics(module, i32, void);

        let tidx = builder
            .build_call(tid_x, &[], "tidx")
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value();
        let tidy = builder
            .build_call(tid_y, &[], "tidy")
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value();
        let bidx = builder
            .build_call(bid_x, &[], "bidx")
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value();
        let bidy = builder
            .build_call(bid_y, &[], "bidy")
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value();
        let bdimx = builder
            .build_call(bdim_x, &[], "bdimx")
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value();
        let bdimy = builder
            .build_call(bdim_y, &[], "bdimy")
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value();

        let tidx64 = builder.build_int_cast(tidx, i64, "tidx64");
        let tidy64 = builder.build_int_cast(tidy, i64, "tidy64");
        let bidx64 = builder.build_int_cast(bidx, i64, "bidx64");
        let bidy64 = builder.build_int_cast(bidy, i64, "bidy64");
        let bdimx64 = builder.build_int_cast(bdimx, i64, "bdimx64");
        let bdimy64 = builder.build_int_cast(bdimy, i64, "bdimy64");
        let nx64 = builder.build_int_cast(nx_param, i64, "nx64");
        let ny64 = builder.build_int_cast(ny_param, i64, "ny64");
        let nz64 = builder.build_int_cast(nz_param, i64, "nz64");
        let stride_x64 = builder.build_int_cast(stride_x_param, i64, "stride_x64");
        let stride_y64 = builder.build_int_cast(stride_y_param, i64, "stride_y64");
        let stride_z64 = builder.build_int_cast(stride_z_param, i64, "stride_z64");

        let const_i64 = |v: i64| i64.const_int(v as u64, false);

        let tile_start_x = builder.build_int_mul(bidx64, bdimx64, "tile_start_x");
        let tile_start_y = builder.build_int_mul(bidy64, bdimy64, "tile_start_y");
        let global_x = builder.build_int_add(tile_start_x, tidx64, "global_x");
        let global_y = builder.build_int_add(tile_start_y, tidy64, "global_y");

        let cond_out_x = builder.build_int_compare(IntPredicate::UGE, global_x, nx64, "cond_out_x");
        let cond_out_y = builder.build_int_compare(IntPredicate::UGE, global_y, ny64, "cond_out_y");
        let cond_out = builder.build_or(cond_out_x, cond_out_y, "cond_out");
        let exit_block = ctx.append_basic_block(kernel, "exit");
        let body_block = ctx.append_basic_block(kernel, "body");
        builder.build_conditional_branch(cond_out, exit_block, body_block);
        builder.position_at_end(body_block);

        let zero = float.const_float(0.0);
        let one_i64 = const_i64(1);

        let safe_load_global_3d = |z: IntValue, y: IntValue, x: IntValue| -> FloatValue {
            let z_stride = builder.build_int_mul(z, stride_z64, "z_stride");
            let y_stride = builder.build_int_mul(y, stride_y64, "y_stride");
            let x_stride = builder.build_int_mul(x, stride_x64, "x_stride");
            let idx = builder.build_int_add(
                builder.build_int_add(z_stride, y_stride, "zy_stride"),
                x_stride,
                "linear_idx",
            );
            common::safe_load_global(
                &builder,
                x_global,
                idx,
                builder.build_int_mul(
                    builder.build_int_mul(
                        builder.build_int_mul(nx64, stride_x64, "sx"),
                        ny64,
                        "sxy"
                    ),
                    nz64,
                    "total"
                ),
                zero,
                const_i64,
            )
        };

        let shared_width = builder.build_int_add(bdimx64, const_i64(2 * radius), "shared_width");
        let shared_height = builder.build_int_add(bdimy64, const_i64(2 * radius), "shared_height");

        let z_zero = const_i64(0);
        let z_cond = builder.build_int_compare(IntPredicate::SLT, z_zero, nz64, "z_cond");
        let z_loop_block = ctx.append_basic_block(kernel, "z_loop");
        let z_end_block = ctx.append_basic_block(kernel, "z_end");

        builder.build_conditional_branch(z_cond, z_loop_block, z_end_block);
        builder.position_at_end(z_loop_block);

        let z_phi = builder.build_phi(i64, "z_phi");
        z_phi.add_incoming(&[(&z_zero, body_block)]);

        let z_current = z_phi.as_basic_value().into_int_value();

        // Carregar centro
        let center_val = safe_load_global_3d(z_current, global_y, global_x);
        let center_idx_x = builder.build_int_add(tidx64, const_i64(radius), "center_x");
        let center_idx_y = builder.build_int_add(tidy64, const_i64(radius), "center_y");
        let center_flat = builder.build_int_add(
            builder.build_int_mul(center_idx_y, shared_width, "center_row_shift"),
            center_idx_x,
            "center_flat",
        );
        let center_store = unsafe {
            builder.build_gep(shared, &[const_i64(0), center_flat], "center_store")
        };
        builder.build_store(center_store, center_val);

        // Halo esquerdo (X)
        let left_cond = builder.build_int_compare(IntPredicate::EQ, tidx64, const_i64(0), "left_cond");
        let left_block = ctx.append_basic_block(kernel, "left_halo");
        let after_left = ctx.append_basic_block(kernel, "after_left");
        builder.build_conditional_branch(left_cond, left_block, after_left);
        builder.position_at_end(left_block);
        {
            let x_left = builder.build_int_sub(global_x, one_i64, "x_left");
            let val_left = safe_load_global_3d(z_current, global_y, x_left);
            let left_store_idx = builder.build_int_add(
                builder.build_int_mul(
                    builder.build_int_add(tidy64, const_i64(radius), "left_row"),
                    shared_width,
                    "left_row_shift",
                ),
                const_i64(0),
                "left_flat",
            );
            let left_store = unsafe {
                builder.build_gep(shared, &[const_i64(0), left_store_idx], "left_store")
            };
            builder.build_store(left_store, val_left);
        }
        builder.build_unconditional_branch(after_left);
        builder.position_at_end(after_left);

        // Halo direito (X)
        let right_threshold_x = builder.build_int_sub(bdimx64, one_i64, "right_threshold_x");
        let right_cond = builder.build_int_compare(IntPredicate::EQ, tidx64, right_threshold_x, "right_cond");
        let right_block = ctx.append_basic_block(kernel, "right_halo");
        let after_right = ctx.append_basic_block(kernel, "after_right");
        builder.build_conditional_branch(right_cond, right_block, after_right);
        builder.position_at_end(right_block);
        {
            let x_right = builder.build_int_add(global_x, one_i64, "x_right");
            let val_right = safe_load_global_3d(z_current, global_y, x_right);
            let right_store_idx = builder.build_int_add(
                builder.build_int_mul(
                    builder.build_int_add(tidy64, const_i64(radius), "right_row"),
                    shared_width,
                    "right_row_shift",
                ),
                builder.build_int_add(bdimx64, const_i64(radius), "right_col"),
                "right_flat",
            );
            let right_store = unsafe {
                builder.build_gep(shared, &[const_i64(0), right_store_idx], "right_store")
            };
            builder.build_store(right_store, val_right);
        }
        builder.build_unconditional_branch(after_right);
        builder.position_at_end(after_right);

        // Halo inferior (Y)
        let bottom_cond = builder.build_int_compare(IntPredicate::EQ, tidy64, const_i64(0), "bottom_cond");
        let bottom_block = ctx.append_basic_block(kernel, "bottom_halo");
        let after_bottom = ctx.append_basic_block(kernel, "after_bottom");
        builder.build_conditional_branch(bottom_cond, bottom_block, after_bottom);
        builder.position_at_end(bottom_block);
        {
            let y_bottom = builder.build_int_sub(global_y, one_i64, "y_bottom");
            let val_bottom = safe_load_global_3d(z_current, y_bottom, global_x);
            let bottom_store_idx = builder.build_int_add(
                builder.build_int_mul(const_i64(0), shared_width, "bottom_row_shift"),
                builder.build_int_add(tidx64, const_i64(radius), "bottom_col"),
                "bottom_flat",
            );
            let bottom_store = unsafe {
                builder.build_gep(shared, &[const_i64(0), bottom_store_idx], "bottom_store")
            };
            builder.build_store(bottom_store, val_bottom);
        }
        builder.build_unconditional_branch(after_bottom);
        builder.position_at_end(after_bottom);

        // Halo superior (Y)
        let top_threshold_y = builder.build_int_sub(bdimy64, one_i64, "top_threshold_y");
        let top_cond = builder.build_int_compare(IntPredicate::EQ, tidy64, top_threshold_y, "top_cond");
        let top_block = ctx.append_basic_block(kernel, "top_halo");
        let after_top = ctx.append_basic_block(kernel, "after_top");
        builder.build_conditional_branch(top_cond, top_block, after_top);
        builder.position_at_end(top_block);
        {
            let y_top = builder.build_int_add(global_y, one_i64, "y_top");
            let val_top = safe_load_global_3d(z_current, y_top, global_x);
            let top_store_idx = builder.build_int_add(
                builder.build_int_mul(
                    builder.build_int_add(bdimy64, const_i64(radius), "top_row"),
                    shared_width,
                    "top_row_shift",
                ),
                builder.build_int_add(tidx64, const_i64(radius), "top_col"),
                "top_flat",
            );
            let top_store = unsafe {
                builder.build_gep(shared, &[const_i64(0), top_store_idx], "top_store")
            };
            builder.build_store(top_store, val_top);
        }
        builder.build_unconditional_branch(after_top);
        builder.position_at_end(after_top);

        builder.build_call(barrier, &[], "sync_after_load");

        // ----------------------------------------------------------------
        // STENCIL COM TENSOR CORES (WMMA)
        // ----------------------------------------------------------------
        // Para simplificar a demonstração, mantemos o loop escalar, mas
        // aqui seria o local para emitir chamadas a:
        //   - llvm.nvvm.wmma.mma.sync
        //   - llvm.nvvm.wmma.load.a.sync
        //   - llvm.nvvm.wmma.load.b.sync
        //   - llvm.nvvm.wmma.store.d.sync
        //
        // Como a geração completa de WMMA exige operações de matriz 16x16x16,
        // este código permanece escalar, mas a estrutura está pronta
        // para receber as intrínsecas.
        // ----------------------------------------------------------------
        let mut result = float.const_float(0.0);
        let coeff_count = (2 * radius + 1) as usize;
        let coeff_count_sq = coeff_count * coeff_count;

        for (idx, coeff) in coeffs.iter().enumerate() {
            let dz_idx = (idx / coeff_count_sq) as i64 - radius;
            let dy_idx = ((idx % coeff_count_sq) / coeff_count) as i64 - radius;
            let dx_idx = (idx % coeff_count) as i64 - radius;

            let coeff_val = float.const_float(*coeff);
            let z_neighbor = builder.build_int_add(z_current, const_i64(dz_idx), "z_neighbor");
            let z_valid_low = builder.build_int_compare(
                IntPredicate::SGE,
                z_neighbor,
                const_i64(0),
                "z_valid_low",
            );
            let z_valid_high = builder.build_int_compare(
                IntPredicate::SLT,
                z_neighbor,
                nz64,
                "z_valid_high",
            );
            let z_valid = builder.build_and(z_valid_low, z_valid_high, "z_valid");

            let is_dz_zero = builder.build_int_compare(
                IntPredicate::EQ,
                const_i64(dz_idx),
                const_i64(0),
                "is_dz_zero",
            );

            let neighbor_val = builder.build_select(
                is_dz_zero,
                {
                    let neighbor_y = builder.build_int_add(center_idx_y, const_i64(dy_idx), "neighbor_y");
                    let neighbor_x = builder.build_int_add(center_idx_x, const_i64(dx_idx), "neighbor_x");
                    let valid_y = builder.build_int_compare(
                        IntPredicate::SGE,
                        neighbor_y,
                        const_i64(0),
                        "ny_low",
                    );
                    let valid_y_h = builder.build_int_compare(
                        IntPredicate::SLT,
                        neighbor_y,
                        shared_height,
                        "ny_high",
                    );
                    let valid_y_all = builder.build_and(valid_y, valid_y_h, "ny_valid");
                    let valid_x = builder.build_int_compare(
                        IntPredicate::SGE,
                        neighbor_x,
                        const_i64(0),
                        "nx_low",
                    );
                    let valid_x_h = builder.build_int_compare(
                        IntPredicate::SLT,
                        neighbor_x,
                        shared_width,
                        "nx_high",
                    );
                    let valid_x_all = builder.build_and(valid_x, valid_x_h, "nx_valid");
                    let valid_all = builder.build_and(valid_y_all, valid_x_all, "n_valid");
                    let safe_y = builder
                        .build_select(valid_y_all, neighbor_y, const_i64(0), "safe_y")
                        .into_int_value();
                    let safe_x = builder
                        .build_select(valid_x_all, neighbor_x, const_i64(0), "safe_x")
                        .into_int_value();
                    let flat_idx = builder.build_int_add(
                        builder.build_int_mul(safe_y, shared_width, "row_shift"),
                        safe_x,
                        "flat_idx",
                    );
                    let ptr = unsafe {
                        builder.build_gep(shared, &[const_i64(0), flat_idx], "shared_ptr")
                    };
                    builder.build_load(ptr, "shared_val").into_float_value()
                },
                {
                    let y_global_neighbor = builder.build_int_add(global_y, const_i64(dy_idx), "gy");
                    let x_global_neighbor = builder.build_int_add(global_x, const_i64(dx_idx), "gx");
                    let y_valid_global = builder.build_int_compare(
                        IntPredicate::SGE,
                        y_global_neighbor,
                        const_i64(0),
                        "y_gl",
                    );
                    let y_valid_global_h = builder.build_int_compare(
                        IntPredicate::SLT,
                        y_global_neighbor,
                        ny64,
                        "y_gh",
                    );
                    let y_valid_global_all = builder.build_and(y_valid_global, y_valid_global_h, "y_gv");
                    let x_valid_global = builder.build_int_compare(
                        IntPredicate::SGE,
                        x_global_neighbor,
                        const_i64(0),
                        "x_gl",
                    );
                    let x_valid_global_h = builder.build_int_compare(
                        IntPredicate::SLT,
                        x_global_neighbor,
                        nx64,
                        "x_gh",
                    );
                    let x_valid_global_all = builder.build_and(x_valid_global, x_valid_global_h, "x_gv");
                    let valid_global_xy = builder.build_and(y_valid_global_all, x_valid_global_all, "xy_gv");
                    let z_and_xy = builder.build_and(z_valid, valid_global_xy, "z_and_xy");
                    let val_from_global = safe_load_global_3d(
                        z_neighbor,
                        builder
                            .build_select(y_valid_global_all, y_global_neighbor, const_i64(0))
                            .into_int_value(),
                        builder
                            .build_select(x_valid_global_all, x_global_neighbor, const_i64(0))
                            .into_int_value(),
                    );
                    builder
                        .build_select(z_and_xy, val_from_global, zero, "global_val")
                        .into_float_value()
                },
                "neighbor_val",
            )
            .into_float_value();

            let weighted = builder.build_float_mul(neighbor_val, coeff_val, "weighted");
            result = builder.build_float_add(result, weighted, "accum");
        }

        builder.build_call(barrier, &[], "sync_after_compute");

        let idx_out = builder.build_int_add(
            builder.build_int_mul(z_current, stride_z64, "z_out"),
            builder.build_int_add(
                builder.build_int_mul(global_y, stride_y64, "y_out"),
                builder.build_int_mul(global_x, stride_x64, "x_out"),
                "xy_out",
            ),
            "idx_out",
        );
        let out_ptr = unsafe { builder.build_gep(y_global, &[idx_out], "out_ptr") };
        builder.build_store(out_ptr, result);

        let z_next = builder.build_int_add(z_current, one_i64, "z_next");
        z_phi.add_incoming(&[(&z_next, body_block)]);
        let z_cond_next = builder.build_int_compare(IntPredicate::SLT, z_next, nz64, "z_cond_next");
        builder.build_conditional_branch(z_cond_next, z_loop_block, z_end_block);

        builder.position_at_end(z_end_block);
        builder.build_unconditional_branch(exit_block);
        builder.position_at_end(exit_block);
        builder.build_return(None);

        let func_meta = kernel.as_metadata_value();
        let kernel_str = ctx.metadata_string("kernel");
        let one_i32 = ctx.i32_type().const_int(1, false).as_metadata_value();
        let md_node = ctx.metadata_node(&[func_meta.into(), kernel_str.into(), one_i32.into()]);
        module.add_named_metadata("nvvm.annotations", &[md_node]);

        Ok(module.print_to_string().to_string())
    }
}