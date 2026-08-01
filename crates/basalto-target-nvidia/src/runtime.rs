// crates/basalto-target-nvidia/src/runtime.rs
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use thiserror::Error;

// ==========================================================================
// Tipos CUDA
// ==========================================================================
pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUresult = u32;
const CUDA_SUCCESS: CUresult = 0;

// ==========================================================================
// Erros
// ==========================================================================
#[derive(Debug, Error)]
pub enum CudaError {
    #[error("Falha ao carregar libcuda.so.1: {0}")]
    LibraryLoad(String),
    #[error("Falha ao obter símbolo '{0}': {1}")]
    MissingSymbol(String, String),
    #[error("CUDA retornou erro código {0} na operação {1}")]
    ApiError(CUresult, String),
    #[error("Erro ao converter string para CString")]
    NulError(#[from] std::ffi::NulError),
}
type Result<T> = std::result::Result<T, CudaError>;

// ==========================================================================
// API CUDA – funções carregadas dinamicamente
// ==========================================================================
pub struct CudaApi {
    _lib: Library,
    cuInit: Symbol<unsafe extern "C" fn(u32) -> CUresult>,
    cuDeviceGet: Symbol<unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult>,
    cuCtxCreate: Symbol<unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult>,
    cuCtxSetCurrent: Symbol<unsafe extern "C" fn(CUcontext) -> CUresult>,
    cuModuleLoadData: Symbol<unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult>,
    cuModuleGetFunction: Symbol<unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult>,
    cuLaunchKernel: Symbol<unsafe extern "C" fn(
        CUfunction,
        u32, u32, u32, // grid dim
        u32, u32, u32, // block dim
        u32,            // sharedMemBytes
        CUstream,       // hStream
        *mut *mut c_void, // kernelParams
        *mut *mut c_void, // extra
    ) -> CUresult>,
    cuCtxSynchronize: Symbol<unsafe extern "C" fn() -> CUresult>,
    cuModuleUnload: Symbol<unsafe extern "C" fn(CUmodule) -> CUresult>,
}

impl CudaApi {
    pub fn new() -> Result<Self> {
        unsafe {
            let lib = Library::new("libcuda.so.1")
                .map_err(|e| CudaError::LibraryLoad(e.to_string()))?;

            let cuInit = lib.get(b"cuInit\0").map_err(|e| CudaError::MissingSymbol("cuInit".into(), e.to_string()))?;
            let cuDeviceGet = lib.get(b"cuDeviceGet\0").map_err(|e| CudaError::MissingSymbol("cuDeviceGet".into(), e.to_string()))?;
            let cuCtxCreate = lib.get(b"cuCtxCreate\0").map_err(|e| CudaError::MissingSymbol("cuCtxCreate".into(), e.to_string()))?;
            let cuCtxSetCurrent = lib.get(b"cuCtxSetCurrent\0").map_err(|e| CudaError::MissingSymbol("cuCtxSetCurrent".into(), e.to_string()))?;
            let cuModuleLoadData = lib.get(b"cuModuleLoadData\0").map_err(|e| CudaError::MissingSymbol("cuModuleLoadData".into(), e.to_string()))?;
            let cuModuleGetFunction = lib.get(b"cuModuleGetFunction\0").map_err(|e| CudaError::MissingSymbol("cuModuleGetFunction".into(), e.to_string()))?;
            let cuLaunchKernel = lib.get(b"cuLaunchKernel\0").map_err(|e| CudaError::MissingSymbol("cuLaunchKernel".into(), e.to_string()))?;
            let cuCtxSynchronize = lib.get(b"cuCtxSynchronize\0").map_err(|e| CudaError::MissingSymbol("cuCtxSynchronize".into(), e.to_string()))?;
            let cuModuleUnload = lib.get(b"cuModuleUnload\0").map_err(|e| CudaError::MissingSymbol("cuModuleUnload".into(), e.to_string()))?;

            Ok(CudaApi {
                _lib: lib,
                cuInit,
                cuDeviceGet,
                cuCtxCreate,
                cuCtxSetCurrent,
                cuModuleLoadData,
                cuModuleGetFunction,
                cuLaunchKernel,
                cuCtxSynchronize,
                cuModuleUnload,
            })
        }
    }

    pub fn init_context(&self) -> Result<CUcontext> {
        unsafe {
            let res = (self.cuInit)(0);
            if res != CUDA_SUCCESS { return Err(CudaError::ApiError(res, "cuInit".into())); }

            let mut device = 0;
            let res = (self.cuDeviceGet)(&mut device, 0);
            if res != CUDA_SUCCESS { return Err(CudaError::ApiError(res, "cuDeviceGet".into())); }

            let mut ctx = ptr::null_mut();
            let res = (self.cuCtxCreate)(&mut ctx, 0, device);
            if res != CUDA_SUCCESS { return Err(CudaError::ApiError(res, "cuCtxCreate".into())); }

            let res = (self.cuCtxSetCurrent)(ctx);
            if res != CUDA_SUCCESS { return Err(CudaError::ApiError(res, "cuCtxSetCurrent".into())); }

            Ok(ctx)
        }
    }

    pub fn load_module(&self, image: &[u8]) -> Result<CUmodule> {
        unsafe {
            let mut module = ptr::null_mut();
            let ptr = image.as_ptr() as *const c_void;
            let res = (self.cuModuleLoadData)(&mut module, ptr);
            if res != CUDA_SUCCESS { return Err(CudaError::ApiError(res, "cuModuleLoadData".into())); }
            Ok(module)
        }
    }

    pub fn get_function(&self, module: CUmodule, name: &str) -> Result<CUfunction> {
        unsafe {
            let cname = CString::new(name)?;
            let mut func = ptr::null_mut();
            let res = (self.cuModuleGetFunction)(&mut func, module, cname.as_ptr());
            if res != CUDA_SUCCESS { return Err(CudaError::ApiError(res, format!("cuModuleGetFunction({})", name))); }
            Ok(func)
        }
    }

    pub fn launch_kernel(
        &self,
        func: CUfunction,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &[*const c_void],
    ) -> Result<()> {
        unsafe {
            let mut params_ptrs: Vec<*mut c_void> = params.iter().map(|p| *p as *mut c_void).collect();
            let kernel_params = params_ptrs.as_mut_ptr();
            let extra = ptr::null_mut();

            let res = (self.cuLaunchKernel)(
                func,
                grid_dim.0, grid_dim.1, grid_dim.2,
                block_dim.0, block_dim.1, block_dim.2,
                shared_mem_bytes,
                ptr::null_mut(), // stream = default
                kernel_params,
                extra,
            );
            if res != CUDA_SUCCESS { return Err(CudaError::ApiError(res, "cuLaunchKernel".into())); }

            let res_sync = (self.cuCtxSynchronize)();
            if res_sync != CUDA_SUCCESS { return Err(CudaError::ApiError(res_sync, "cuCtxSynchronize".into())); }

            Ok(())
        }
    }

    pub fn unload_module(&self, module: CUmodule) -> Result<()> {
        unsafe {
            let res = (self.cuModuleUnload)(module);
            if res != CUDA_SUCCESS { return Err(CudaError::ApiError(res, "cuModuleUnload".into())); }
            Ok(())
        }
    }
}

// ==========================================================================
// Runtime de alto nível com suporte a shared_mem
// ==========================================================================
pub struct NvidiaRuntime {
    api: CudaApi,
    ctx: CUcontext,
}

impl NvidiaRuntime {
    pub fn new() -> Result<Self> {
        let api = CudaApi::new()?;
        let ctx = api.init_context()?;
        Ok(Self { api, ctx })
    }

    pub fn launch(
        &self,
        module_data: &[u8],
        function_name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &[*const c_void],
    ) -> Result<()> {
        let module = self.api.load_module(module_data)?;
        let func = self.api.get_function(module, function_name)?;
        let result = self.api.launch_kernel(func, grid, block, shared_mem_bytes, params);
        let _ = self.api.unload_module(module);
        result
    }
}