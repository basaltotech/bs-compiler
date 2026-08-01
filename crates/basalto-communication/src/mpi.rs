use libloading::{Library, Symbol};
use std::ffi::{c_int, c_void, CString};
use anyhow::{anyhow, Result};

pub struct MpiRuntime {
    _lib: Library,
    mpi_comm_rank: Symbol<unsafe extern "C" fn(c_int, *mut c_int) -> c_int>,
    mpi_comm_size: Symbol<unsafe extern "C" fn(c_int, *mut c_int) -> c_int>,
    mpi_sendrecv: Symbol<
        unsafe extern "C" fn(
            *const c_void,
            c_int,
            c_int,
            c_int,
            *mut c_void,
            c_int,
            c_int,
            c_int,
            *mut c_int,
        ) -> c_int,
    >,
    mpi_barrier: Symbol<unsafe extern "C" fn(c_int) -> c_int>,
    mpi_finalize: Symbol<unsafe extern "C" fn() -> c_int>,
}

impl MpiRuntime {
    pub fn new() -> Result<Self> {
        unsafe {
            let lib = Library::new("libmpi.so").ok();
            let lib = match lib {
                Some(l) => l,
                None => {
                    // Fallback para libmpich ou openmpi
                    Library::new("libmpich.so")
                        .or_else(|_| Library::new("libopenmpi.so"))
                        .map_err(|e| anyhow!("Nenhuma biblioteca MPI encontrada: {}", e))?
                }
            };

            let mpi_comm_rank = lib.get(b"MPI_Comm_rank\0")
                .map_err(|e| anyhow!("MPI_Comm_rank não encontrado: {}", e))?;
            let mpi_comm_size = lib.get(b"MPI_Comm_size\0")
                .map_err(|e| anyhow!("MPI_Comm_size não encontrado: {}", e))?;
            let mpi_sendrecv = lib.get(b"MPI_Sendrecv\0")
                .map_err(|e| anyhow!("MPI_Sendrecv não encontrado: {}", e))?;
            let mpi_barrier = lib.get(b"MPI_Barrier\0")
                .map_err(|e| anyhow!("MPI_Barrier não encontrado: {}", e))?;
            let mpi_finalize = lib.get(b"MPI_Finalize\0")
                .map_err(|e| anyhow!("MPI_Finalize não encontrado: {}", e))?;

            // Inicializa MPI (assumindo que já foi inicializado pelo processo pai)
            // Em produção, isso viria do Slurm/MPI launcher.

            Ok(Self {
                _lib: lib,
                mpi_comm_rank,
                mpi_comm_size,
                mpi_sendrecv,
                mpi_barrier,
                mpi_finalize,
            })
        }
    }

    pub fn rank(&self) -> Result<i32> {
        unsafe {
            let mut rank = 0;
            let res = (self.mpi_comm_rank)(0, &mut rank);
            if res != 0 {
                return Err(anyhow!("MPI_Comm_rank falhou com código {}", res));
            }
            Ok(rank)
        }
    }

    pub fn size(&self) -> Result<i32> {
        unsafe {
            let mut size = 0;
            let res = (self.mpi_comm_size)(0, &mut size);
            if res != 0 {
                return Err(anyhow!("MPI_Comm_size falhou com código {}", res));
            }
            Ok(size)
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
            let res = (self.mpi_sendrecv)(
                send_buf,
                send_count,
                0, // MPI_DOUBLE (placeholder)
                dest,
                0, // tag
                recv_buf,
                recv_count,
                0, // MPI_DOUBLE
                source,
                0, // tag
                &mut status,
            );
            if res != 0 {
                return Err(anyhow!("MPI_Sendrecv falhou com código {}", res));
            }
            Ok(())
        }
    }

    pub fn barrier(&self) -> Result<()> {
        unsafe {
            let res = (self.mpi_barrier)(0);
            if res != 0 {
                return Err(anyhow!("MPI_Barrier falhou com código {}", res));
            }
            Ok(())
        }
    }
}

impl Drop for MpiRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.mpi_finalize)();
        }
    }
}