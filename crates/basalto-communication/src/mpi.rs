//! Wrapper para MPI (Message Passing Interface).
//! Carregado dinamicamente – suporta libmpi.so, libmpich.so, libopenmpi.so.

use libloading::{Library, Symbol};
use std::ffi::{c_int, c_void};
use anyhow::{anyhow, Result};

pub struct MpiRuntime {
    _lib: Library,
    pub mpi_init: Symbol<unsafe extern "C" fn(*mut c_int, *mut *mut *mut c_void) -> c_int>,
    pub mpi_finalize: Symbol<unsafe extern "C" fn() -> c_int>,
    pub mpi_comm_rank: Symbol<unsafe extern "C" fn(c_int, *mut c_int) -> c_int>,
    pub mpi_comm_size: Symbol<unsafe extern "C" fn(c_int, *mut c_int) -> c_int>,
    pub mpi_send: Symbol<unsafe extern "C" fn(*const c_void, c_int, c_int, c_int, c_int) -> c_int>,
    pub mpi_recv: Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int, c_int, c_int, *mut c_int) -> c_int>,
    pub mpi_sendrecv: Symbol<unsafe extern "C" fn(
        *const c_void, c_int, c_int, c_int,
        *mut c_void, c_int, c_int, c_int,
        *mut c_int,
    ) -> c_int>,
    pub mpi_barrier: Symbol<unsafe extern "C" fn(c_int) -> c_int>,
    pub mpi_initialized: Symbol<unsafe extern "C" fn(*mut c_int) -> c_int>,
    pub initialized: bool,
}

impl MpiRuntime {
    pub fn new() -> Result<Self> {
        unsafe {
            // Tenta carregar diferentes bibliotecas MPI
            let lib = Library::new("libmpi.so")
                .or_else(|_| Library::new("libmpich.so"))
                .or_else(|_| Library::new("libopenmpi.so"))
                .map_err(|e| anyhow!("Nenhuma biblioteca MPI encontrada: {}", e))?;

            let mpi_init = lib.get(b"MPI_Init\0")
                .map_err(|e| anyhow!("MPI_Init não encontrado: {}", e))?;
            let mpi_finalize = lib.get(b"MPI_Finalize\0")
                .map_err(|e| anyhow!("MPI_Finalize não encontrado: {}", e))?;
            let mpi_comm_rank = lib.get(b"MPI_Comm_rank\0")
                .map_err(|e| anyhow!("MPI_Comm_rank não encontrado: {}", e))?;
            let mpi_comm_size = lib.get(b"MPI_Comm_size\0")
                .map_err(|e| anyhow!("MPI_Comm_size não encontrado: {}", e))?;
            let mpi_send = lib.get(b"MPI_Send\0")
                .map_err(|e| anyhow!("MPI_Send não encontrado: {}", e))?;
            let mpi_recv = lib.get(b"MPI_Recv\0")
                .map_err(|e| anyhow!("MPI_Recv não encontrado: {}", e))?;
            let mpi_sendrecv = lib.get(b"MPI_Sendrecv\0")
                .map_err(|e| anyhow!("MPI_Sendrecv não encontrado: {}", e))?;
            let mpi_barrier = lib.get(b"MPI_Barrier\0")
                .map_err(|e| anyhow!("MPI_Barrier não encontrado: {}", e))?;
            let mpi_initialized = lib.get(b"MPI_Initialized\0")
                .map_err(|e| anyhow!("MPI_Initialized não encontrado: {}", e))?;

            // Verifica se MPI já foi inicializado (ex: pelo Slurm/MPI launcher)
            let mut flag = 0;
            let _ = mpi_initialized(&mut flag);
            let initialized = flag != 0;

            if !initialized {
                // Inicializa MPI com argumentos vazios
                let mut argc = 0;
                let mut argv = std::ptr::null_mut();
                let ret = mpi_init(&mut argc, &mut argv);
                if ret != 0 {
                    return Err(anyhow!("MPI_Init falhou com código {}", ret));
                }
            }

            Ok(Self {
                _lib: lib,
                mpi_init,
                mpi_finalize,
                mpi_comm_rank,
                mpi_comm_size,
                mpi_send,
                mpi_recv,
                mpi_sendrecv,
                mpi_barrier,
                mpi_initialized,
                initialized,
            })
        }
    }

    pub fn rank(&self) -> Result<i32> {
        unsafe {
            let mut rank = 0;
            let ret = (self.mpi_comm_rank)(0, &mut rank);
            if ret != 0 {
                return Err(anyhow!("MPI_Comm_rank falhou com código {}", ret));
            }
            Ok(rank)
        }
    }

    pub fn size(&self) -> Result<i32> {
        unsafe {
            let mut size = 0;
            let ret = (self.mpi_comm_size)(0, &mut size);
            if ret != 0 {
                return Err(anyhow!("MPI_Comm_size falhou com código {}", ret));
            }
            Ok(size)
        }
    }

    pub fn barrier(&self) -> Result<()> {
        unsafe {
            let ret = (self.mpi_barrier)(0);
            if ret != 0 {
                return Err(anyhow!("MPI_Barrier falhou com código {}", ret));
            }
            Ok(())
        }
    }

    pub fn sendrecv(
        &self,
        send_buf: *const c_void,
        send_count: i32,
        dest: i32,
        recv_buf: *mut c_void,
        recv_count: i32,
        source: i32,
    ) -> Result<()> {
        unsafe {
            let mut status = 0;
            let ret = (self.mpi_sendrecv)(
                send_buf, send_count, 0, dest,
                recv_buf, recv_count, 0, source,
                &mut status,
            );
            if ret != 0 {
                return Err(anyhow!("MPI_Sendrecv falhou com código {}", ret));
            }
            Ok(())
        }
    }
}

impl Drop for MpiRuntime {
    fn drop(&mut self) {
        if !self.initialized {
            unsafe {
                let _ = (self.mpi_finalize)();
            }
        }
    }
}