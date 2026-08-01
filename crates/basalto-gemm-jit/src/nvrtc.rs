s//! Wrapper para NVRTC (NVIDIA Runtime Compilation) – compila código CUDA em tempo real.
//! Carregado dinamicamente via libloading.

use libloading::{Library, Symbol};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;
use anyhow::{anyhow, Result};

pub type nvrtcResult = c_int;
pub type nvrtcProgram = *mut c_void;

pub const NVRTC_SUCCESS: nvrtcResult = 0;
pub const NVRTC_ERROR_COMPILATION: nvrtcResult = 6;

pub struct NvrtcRuntime {
    _lib: Library,
    pub nvrtc_create_program: Symbol<unsafe extern "C" fn(
        *mut nvrtcProgram,
        *const *const c_char,
        c_int,
        *const *const c_char,
        c_int,
    ) -> nvrtcResult>,
    pub nvrtc_destroy_program: Symbol<unsafe extern "C" fn(nvrtcProgram) -> nvrtcResult>,
    pub nvrtc_compile_program: Symbol<unsafe extern "C" fn(nvrtcProgram, *const *const c_char) -> nvrtcResult>,
    pub nvrtc_get_ptx: Symbol<unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> nvrtcResult>,
    pub nvrtc_get_ptx_size: Symbol<unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult>,
    pub nvrtc_get_error_log: Symbol<unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> nvrtcResult>,
    pub nvrtc_get_error_log_size: Symbol<unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult>,
}

impl NvrtcRuntime {
    pub fn new() -> Result<Self> {
        unsafe {
            let lib = Library::new("libnvrtc.so")
                .or_else(|_| Library::new("libnvrtc.so.12"))
                .or_else(|_| Library::new("libnvrtc.so.11"))
                .map_err(|e| anyhow!("Falha ao carregar libnvrtc: {}", e))?;

            let nvrtc_create_program = lib.get(b"nvrtcCreateProgram\0")
                .map_err(|e| anyhow!("nvrtcCreateProgram não encontrado: {}", e))?;
            let nvrtc_destroy_program = lib.get(b"nvrtcDestroyProgram\0")
                .map_err(|e| anyhow!("nvrtcDestroyProgram não encontrado: {}", e))?;
            let nvrtc_compile_program = lib.get(b"nvrtcCompileProgram\0")
                .map_err(|e| anyhow!("nvrtcCompileProgram não encontrado: {}", e))?;
            let nvrtc_get_ptx = lib.get(b"nvrtcGetPTX\0")
                .map_err(|e| anyhow!("nvrtcGetPTX não encontrado: {}", e))?;
            let nvrtc_get_ptx_size = lib.get(b"nvrtcGetPTXSize\0")
                .map_err(|e| anyhow!("nvrtcGetPTXSize não encontrado: {}", e))?;
            let nvrtc_get_error_log = lib.get(b"nvrtcGetErrorLog\0")
                .map_err(|e| anyhow!("nvrtcGetErrorLog não encontrado: {}", e))?;
            let nvrtc_get_error_log_size = lib.get(b"nvrtcGetErrorLogSize\0")
                .map_err(|e| anyhow!("nvrtcGetErrorLogSize não encontrado: {}", e))?;

            Ok(Self {
                _lib: lib,
                nvrtc_create_program,
                nvrtc_destroy_program,
                nvrtc_compile_program,
                nvrtc_get_ptx,
                nvrtc_get_ptx_size,
                nvrtc_get_error_log,
                nvrtc_get_error_log_size,
            })
        }
    }

    /// Compila código CUDA para PTX.
    /// `source` é o código fonte CUDA, `name` é o nome do programa (para debug).
    /// `arch` é a arquitetura alvo (ex: "sm_80").
    pub fn compile_to_ptx(&self, source: &str, name: &str, arch: &str) -> Result<Vec<u8>> {
        unsafe {
            let source_cstr = CString::new(source)?;
            let name_cstr = CString::new(name)?;
            let sources = [source_cstr.as_ptr()];
            let names = [name_cstr.as_ptr()];

            let mut program: nvrtcProgram = ptr::null_mut();
            let status = (self.nvrtc_create_program)(
                &mut program,
                sources.as_ptr(),
                1,
                names.as_ptr(),
                0,
            );
            if status != NVRTC_SUCCESS {
                return Err(anyhow!("nvrtcCreateProgram falhou com status {}", status));
            }

            // Opções de compilação
            let compile_args = [
                format!("--gpu-architecture={}", arch),
                "--std=c++17".to_string(),
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1".to_string(),
            ];
            let arg_ptrs: Vec<CString> = compile_args
                .iter()
                .map(|s| CString::new(s.as_str()).unwrap())
                .collect();
            let arg_ptrs_raw: Vec<*const c_char> = arg_ptrs.iter().map(|s| s.as_ptr()).collect();

            let status = (self.nvrtc_compile_program)(program, arg_ptrs_raw.as_ptr());
            if status != NVRTC_SUCCESS {
                // Pega o log de erro
                let mut log_size = 0;
                let _ = (self.nvrtc_get_error_log_size)(program, &mut log_size);
                if log_size > 0 {
                    let mut log_buf = vec![0u8; log_size + 1];
                    let _ = (self.nvrtc_get_error_log)(program, log_buf.as_mut_ptr() as *mut c_char);
                    let log = CStr::from_ptr(log_buf.as_ptr() as *const c_char)
                        .to_string_lossy()
                        .to_string();
                    let _ = (self.nvrtc_destroy_program)(program);
                    return Err(anyhow!("NVRTC compilação falhou: {}", log));
                }
                let _ = (self.nvrtc_destroy_program)(program);
                return Err(anyhow!("NVRTC compilação falhou com status {}", status));
            }

            // Pega o PTX
            let mut ptx_size = 0;
            let status = (self.nvrtc_get_ptx_size)(program, &mut ptx_size);
            if status != NVRTC_SUCCESS {
                let _ = (self.nvrtc_destroy_program)(program);
                return Err(anyhow!("nvrtcGetPTXSize falhou com status {}", status));
            }

            let mut ptx_buf = vec![0u8; ptx_size + 1];
            let status = (self.nvrtc_get_ptx)(program, ptx_buf.as_mut_ptr() as *mut c_char);
            let _ = (self.nvrtc_destroy_program)(program);

            if status != NVRTC_SUCCESS {
                return Err(anyhow!("nvrtcGetPTX falhou com status {}", status));
            }

            // Remove o byte nulo extra
            ptx_buf.truncate(ptx_size);
            Ok(ptx_buf)
        }
    }
}