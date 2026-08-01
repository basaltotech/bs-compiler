use libloading::{Library, Symbol};
use std::ffi::{c_int, c_void, c_float, c_double};
use anyhow::{anyhow, Result};

pub type cublasHandle_t = *mut c_void;
pub type cublasStatus_t = c_int;
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

pub const CUBLAS_OP_N: c_int = 0;
pub const CUBLAS_OP_T: c_int = 1;

pub struct CublasRuntime {
    _lib: Library,
    pub cublas_create: Symbol<unsafe extern "C" fn(*mut cublasHandle_t) -> cublasStatus_t>,
    pub cublas_destroy: Symbol<unsafe extern "C" fn(cublasHandle_t) -> cublasStatus_t>,
    pub cublas_sgemm: Symbol<unsafe extern "C" fn(
        cublasHandle_t, c_int, c_int, c_int, c_int, c_int,
        *const c_float, *const c_void, c_int, *const c_void, c_int,
        *const c_float, *mut c_void, c_int,
    ) -> cublasStatus_t>,
    pub cublas_dgemm: Symbol<unsafe extern "C" fn(
        cublasHandle_t, c_int, c_int, c_int, c_int, c_int,
        *const c_double, *const c_void, c_int, *const c_void, c_int,
        *const c_double, *mut c_void, c_int,
    ) -> cublasStatus_t>,
}

impl CublasRuntime {
    pub fn new() -> Result<Self> {
        unsafe {
            let lib = Library::new("libcublas.so")
                .or_else(|_| Library::new("libcublas.so.12"))
                .or_else(|_| Library::new("libcublas.so.11"))
                .map_err(|e| anyhow!("Falha ao carregar libcublas: {}", e))?;

            let cublas_create = lib.get(b"cublasCreate_v2\0")
                .map_err(|e| anyhow!("cublasCreate_v2 não encontrado: {}", e))?;
            let cublas_destroy = lib.get(b"cublasDestroy_v2\0")
                .map_err(|e| anyhow!("cublasDestroy_v2 não encontrado: {}", e))?;
            let cublas_sgemm = lib.get(b"cublasSgemm_v2\0")
                .map_err(|e| anyhow!("cublasSgemm_v2 não encontrado: {}", e))?;
            let cublas_dgemm = lib.get(b"cublasDgemm_v2\0")
                .map_err(|e| anyhow!("cublasDgemm_v2 não encontrado: {}", e))?;

            Ok(Self {
                _lib: lib,
                cublas_create,
                cublas_destroy,
                cublas_sgemm,
                cublas_dgemm,
            })
        }
    }

    pub fn create_handle(&self) -> Result<cublasHandle_t> {
        unsafe {
            let mut handle = std::ptr::null_mut();
            let status = (self.cublas_create)(&mut handle);
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(anyhow!("cublasCreate falhou com status {}", status));
            }
            Ok(handle)
        }
    }

    pub fn destroy_handle(&self, handle: cublasHandle_t) -> Result<()> {
        unsafe {
            let status = (self.cublas_destroy)(handle);
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(anyhow!("cublasDestroy falhou com status {}", status));
            }
            Ok(())
        }
    }
}