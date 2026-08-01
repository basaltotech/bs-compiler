use libloading::{Library, Symbol};
use std::ffi::{c_int, c_void};
use anyhow::{anyhow, Result};

pub struct NcclRuntime {
    _lib: Library,
    nccl_all_reduce: Symbol<
        unsafe extern "C" fn(
            *const c_void,
            *mut c_void,
            usize,
            c_int,
            c_int,
            c_int,
            *mut c_void,
        ) -> c_int,
    >,
    nccl_broadcast: Symbol<
        unsafe extern "C" fn(
            *const c_void,
            *mut c_void,
            usize,
            c_int,
            c_int,
            c_int,
            *mut c_void,
        ) -> c_int,
    >,
}

impl NcclRuntime {
    pub fn new() -> Result<Self> {
        unsafe {
            let lib = Library::new("libnccl.so")
                .map_err(|e| anyhow!("Falha ao carregar libnccl.so: {}", e))?;

            let nccl_all_reduce = lib.get(b"ncclAllReduce\0")
                .map_err(|e| anyhow!("ncclAllReduce não encontrado: {}", e))?;
            let nccl_broadcast = lib.get(b"ncclBroadcast\0")
                .map_err(|e| anyhow!("ncclBroadcast não encontrado: {}", e))?;

            Ok(Self {
                _lib: lib,
                nccl_all_reduce,
                nccl_broadcast,
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
            let res = (self.nccl_all_reduce)(send_buf, recv_buf, count, data_type, op, comm, stream);
            if res != 0 {
                return Err(anyhow!("ncclAllReduce falhou com código {}", res));
            }
            Ok(())
        }
    }
}