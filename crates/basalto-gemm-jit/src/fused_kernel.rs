//! Execução de kernels fundidos (CUTLASS JIT) a partir do executor.

use anyhow::{anyhow, Result};
use std::ffi::c_void;
use std::sync::Arc;
use crate::cutlass::{CutlassJit, FusedOp, GemmConfig};
use crate::nvrtc::NvrtcRuntime;

/// Executa um kernel GEMM fundido usando CUTLASS JIT.
/// # Parâmetros
/// - `a_ptr`, `b_ptr`, `c_ptr`: ponteiros de dispositivo para as matrizes
/// - `m`, `n`, `k`: dimensões (C = A * B)
/// - `trans_a`, `trans_b`: se as matrizes de entrada estão transpostas
/// - `batch`: número de matrizes em um lote
/// - `dtype`: "f32" ou "f64"
/// - `fused_op`: operação de fusão (Bias, Relu, etc.)
/// - `arch`: arquitetura alvo (ex: "sm_80")
pub fn execute_fused_gemm(
    a_ptr: *mut c_void,
    b_ptr: *mut c_void,
    c_ptr: *mut c_void,
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
    batch: usize,
    dtype: &str,
    fused_op: FusedOp,
    arch: &str,
) -> Result<Vec<u8>> {
    let nvrtc = Arc::new(NvrtcRuntime::new()?);
    let mut jit = CutlassJit::new(nvrtc);

    let config = GemmConfig {
        m,
        n,
        k,
        batch,
        dtype: dtype.to_string(),
        trans_a,
        trans_b,
        fused_op,
        arch: arch.to_string(),
    };

    let ptx = jit.compile(&config)?;

    // TODO: carregar o PTX via cuModuleLoadData e executar o kernel
    // Por enquanto, retorna o PTX para ser carregado pelo executor.
    Ok(ptx)
}