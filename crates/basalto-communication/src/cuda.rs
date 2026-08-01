//! Wrapper para funções essenciais da CUDA Runtime API.
//! Carregado dinamicamente via libloading.

use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::ptr;
use anyhow::{anyhow, Result};

/// Constantes cudaMemcpyKind
pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

pub struct CudaRuntime {
    _lib: Library,
    pub cuda_memcpy: Symbol<unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32>,
    pub cuda_malloc_host: Symbol<unsafe extern "C" fn(*mut *mut c_void, usize) -> i32>,
    pub cuda_free_host: Symbol<unsafe extern "C" fn(*mut c_void) -> i32>,
    pub cuda_get_device: Symbol<unsafe extern "C" fn(*mut i32) -> i32>,
    pub cuda_set_device: Symbol<unsafe extern "C" fn(i32) -> i32>,
}

impl CudaRuntime {
    pub fn new() -> Result<Self> {
        unsafe {
            let lib = Library::new("libcudart.so")
                .or_else(|_| Library::new("libcudart.so.12"))
                .or_else(|_| Library::new("libcudart.so.11"))
                .map_err(|e| anyhow!("Falha ao carregar libcudart: {}", e))?;

            let cuda_memcpy = lib.get(b"cudaMemcpy\0")
                .map_err(|e| anyhow!("cudaMemcpy não encontrado: {}", e))?;
            let cuda_malloc_host = lib.get(b"cudaMallocHost\0")
                .map_err(|e| anyhow!("cudaMallocHost não encontrado: {}", e))?;
            let cuda_free_host = lib.get(b"cudaFreeHost\0")
                .map_err(|e| anyhow!("cudaFreeHost não encontrado: {}", e))?;
            let cuda_get_device = lib.get(b"cudaGetDevice\0")
                .map_err(|e| anyhow!("cudaGetDevice não encontrado: {}", e))?;
            let cuda_set_device = lib.get(b"cudaSetDevice\0")
                .map_err(|e| anyhow!("cudaSetDevice não encontrado: {}", e))?;

            Ok(Self {
                _lib: lib,
                cuda_memcpy,
                cuda_malloc_host,
                cuda_free_host,
                cuda_get_device,
                cuda_set_device,
            })
        }
    }

    /// Copia dados entre GPU e CPU (ou GPU↔GPU).
    pub unsafe fn memcpy(&self, dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> Result<()> {
        let ret = (self.cuda_memcpy)(dst, src, bytes, kind);
        if ret != 0 {
            return Err(anyhow!("cudaMemcpy falhou com código {}", ret));
        }
        Ok(())
    }

    /// Aloca memória pinned (page-locked) na CPU – acelera transferências GPU↔CPU.
    pub unsafe fn malloc_host(&self, bytes: usize) -> Result<*mut c_void> {
        let mut ptr = ptr::null_mut();
        let ret = (self.cuda_malloc_host)(&mut ptr, bytes);
        if ret != 0 {
            return Err(anyhow!("cudaMallocHost falhou com código {}", ret));
        }
        Ok(ptr)
    }

    /// Libera memória pinned alocada com malloc_host.
    pub unsafe fn free_host(&self, ptr: *mut c_void) -> Result<()> {
        let ret = (self.cuda_free_host)(ptr);
        if ret != 0 {
            return Err(anyhow!("cudaFreeHost falhou com código {}", ret));
        }
        Ok(())
    }

    /// Obtém o device CUDA atual.
    pub unsafe fn get_device(&self) -> Result<i32> {
        let mut dev = 0;
        let ret = (self.cuda_get_device)(&mut dev);
        if ret != 0 {
            return Err(anyhow!("cudaGetDevice falhou com código {}", ret));
        }
        Ok(dev)
    }

    /// Define o device CUDA ativo.
    pub unsafe fn set_device(&self, dev: i32) -> Result<()> {
        let ret = (self.cuda_set_device)(dev);
        if ret != 0 {
            return Err(anyhow!("cudaSetDevice falhou com código {}", ret));
        }
        Ok(())
    }
}