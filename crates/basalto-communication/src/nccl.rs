//! Wrapper para NCCL (NVIDIA Collective Communications Library).
//! Carregado dinamicamente – suporta libnccl.so, libnccl.so.2.

use libloading::{Library, Symbol};
use std::ffi::{c_int, c_void};
use anyhow::{anyhow, Result};

/// Tipos de dados NCCL
pub const NCCL_FLOAT: c_int = 0;
pub const NCCL_FLOAT16: c_int = 1;
pub const NCCL_DOUBLE: c_int = 2;
pub const NCCL_INT32: c_int = 3;
pub const NCCL_INT64: c_int = 4;

/// Operações de redução NCCL
pub const NCCL_SUM: c_int = 0;
pub const NCCL_PROD: c_int = 1;
pub const NCCL_MAX: c_int = 2;
pub const NCCL_MIN: c_int = 3;

pub struct NcclRuntime {
    _lib: Library,
    pub nccl_init: Symbol<unsafe extern "C" fn() -> c_int>,
    pub nccl_finalize: Symbol<unsafe extern "C" fn() -> c_int>,
    pub nccl_all_reduce: Symbol<unsafe extern "C" fn(
        *const c_void, *mut c_void, usize, c_int, c_int, c_int, *mut c_void,
    ) -> c_int>,
    pub nccl_broadcast: Symbol<unsafe extern "C" fn(
        *const c_void, *mut c_void, usize, c_int, c_int, c_int, *mut c_void,
    ) -> c_int>,
    pub nccl_send: Symbol<unsafe extern "C" fn(
        *const c_void, usize, c_int, c_int, *mut c_void,
    ) -> c_int>,
    pub nccl_recv: Symbol<unsafe extern "C" fn(
        *mut c_void, usize, c_int, c_int, *mut c_void,
    ) -> c_int>,
    pub nccl_group_start: Symbol<unsafe extern "C" fn() -> c_int>,
    pub nccl_group_end: Symbol<unsafe extern "C" fn() -> c_int>,
    pub initialized: bool,
}

impl NcclRuntime {
    pub fn new() -> Result<Self> {
        unsafe {
            let lib = Library::new("libnccl.so")
                .or_else(|_| Library::new("libnccl.so.2"))
                .map_err(|e| anyhow!("NCCL não encontrado: {}", e))?;

            let nccl_init = lib.get(b"ncclInit\0")
                .map_err(|e| anyhow!("ncclInit não encontrado: {}", e))?;
            let nccl_finalize = lib.get(b"ncclFinalize\0")
                .map_err(|e| anyhow!("ncclFinalize não encontrado: {}", e))?;
            let nccl_all_reduce = lib.get(b"ncclAllReduce\0")
                .map_err(|e| anyhow!("ncclAllReduce não encontrado: {}", e))?;
            let nccl_broadcast = lib.get(b"ncclBroadcast\0")
                .map_err(|e| anyhow!("ncclBroadcast não encontrado: {}", e))?;
            let nccl_send = lib.get(b"ncclSend\0")
                .map_err(|e| anyhow!("ncclSend não encontrado: {}", e))?;
            let nccl_recv = lib.get(b"ncclRecv\0")
                .map_err(|e| anyhow!("ncclRecv não encontrado: {}", e))?;
            let nccl_group_start = lib.get(b"ncclGroupStart\0")
                .map_err(|e| anyhow!("ncclGroupStart não encontrado: {}", e))?;
            let nccl_group_end = lib.get(b"ncclGroupEnd\0")
                .map_err(|e| anyhow!("ncclGroupEnd não encontrado: {}", e))?;

            // Inicializa NCCL
            let ret = nccl_init();
            if ret != 0 {
                return Err(anyhow!("ncclInit falhou com código {}", ret));
            }

            Ok(Self {
                _lib: lib,
                nccl_init,
                nccl_finalize,
                nccl_all_reduce,
                nccl_broadcast,
                nccl_send,
                nccl_recv,
                nccl_group_start,
                nccl_group_end,
                initialized: true,
            })
        }
    }

    pub fn all_reduce(
        &self,
        send_buf: *const c_void,
        recv_buf: *mut c_void,
        count: usize,
        data_type: c_int,
        op: c_int,
        comm: c_int,
        stream: *mut c_void,
    ) -> Result<()> {
        unsafe {
            let ret = (self.nccl_all_reduce)(send_buf, recv_buf, count, data_type, op, comm, stream);
            if ret != 0 {
                return Err(anyhow!("ncclAllReduce falhou com código {}", ret));
            }
            Ok(())
        }
    }

    pub fn send(
        &self,
        send_buf: *const c_void,
        count: usize,
        data_type: c_int,
        peer: c_int,
        stream: *mut c_void,
    ) -> Result<()> {
        unsafe {
            let ret = (self.nccl_send)(send_buf, count, data_type, peer, stream);
            if ret != 0 {
                return Err(anyhow!("ncclSend falhou com código {}", ret));
            }
            Ok(())
        }
    }

    pub fn recv(
        &self,
        recv_buf: *mut c_void,
        count: usize,
        data_type: c_int,
        peer: c_int,
        stream: *mut c_void,
    ) -> Result<()> {
        unsafe {
            let ret = (self.nccl_recv)(recv_buf, count, data_type, peer, stream);
            if ret != 0 {
                return Err(anyhow!("ncclRecv falhou com código {}", ret));
            }
            Ok(())
        }
    }

    pub fn group_start(&self) -> Result<()> {
        unsafe {
            let ret = (self.nccl_group_start)();
            if ret != 0 {
                return Err(anyhow!("ncclGroupStart falhou com código {}", ret));
            }
            Ok(())
        }
    }

    pub fn group_end(&self) -> Result<()> {
        unsafe {
            let ret = (self.nccl_group_end)();
            if ret != 0 {
                return Err(anyhow!("ncclGroupEnd falhou com código {}", ret));
            }
            Ok(())
        }
    }
}

impl Drop for NcclRuntime {
    fn drop(&mut self) {
        if self.initialized {
            unsafe {
                let _ = (self.nccl_finalize)();
            }
        }
    }
}