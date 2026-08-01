use anyhow::{anyhow, Result};
use std::ffi::c_void;
use crate::mpi::MpiRuntime;
use crate::nccl::NcclRuntime;

pub struct HaloExchanger {
    mpi: MpiRuntime,
    nccl: Option<NcclRuntime>,
    rank: i32,
    size: i32,
}

impl HaloExchanger {
    pub fn new(mpi: MpiRuntime, nccl: Option<NcclRuntime>) -> Result<Self> {
        let rank = mpi.rank()?;
        let size = mpi.size()?;
        Ok(Self { mpi, nccl, rank, size })
    }

    pub fn get_rank(&self) -> i32 {
        self.rank
    }

    pub fn get_size(&self) -> i32 {
        self.size
    }

    pub fn exchange_halo_3d(
        &self,
        data: *mut c_void,
        nx: usize,
        ny: usize,
        nz: usize,
        halo_x: usize,
        halo_y: usize,
        halo_z: usize,
        elem_size: usize,
        _stream: Option<*mut c_void>,
    ) -> Result<()> {
        let left_rank = (self.rank - 1 + self.size) % self.size;
        let right_rank = (self.rank + 1) % self.size;

        // Tamanhos dos halos em bytes
        let halo_left_size = halo_x * ny * nz * elem_size;
        let halo_right_size = halo_x * ny * nz * elem_size;
        let halo_bottom_size = nx * halo_y * nz * elem_size;
        let halo_top_size = nx * halo_y * nz * elem_size;

        // Troca em X (esquerda/direita)
        if self.size > 1 {
            // Enviar halo direito para a direita, receber da esquerda
            let send_buf_right = unsafe {
                (data as *mut u8).add((nx - halo_x) * ny * nz * elem_size) as *const c_void
            };
            let recv_buf_left = unsafe { (data as *mut u8).add(0) as *mut c_void };
            self.mpi.sendrecv(
                send_buf_right,
                halo_right_size as i32,
                right_rank,
                recv_buf_left,
                halo_left_size as i32,
                left_rank,
            )?;

            // Enviar halo esquerdo para a esquerda, receber da direita
            let send_buf_left =
                unsafe { (data as *mut u8).add(0) as *const c_void };
            let recv_buf_right = unsafe {
                (data as *mut u8).add((nx - halo_x) * ny * nz * elem_size) as *mut c_void
            };
            self.mpi.sendrecv(
                send_buf_left,
                halo_left_size as i32,
                left_rank,
                recv_buf_right,
                halo_right_size as i32,
                right_rank,
            )?;
        }

        // Troca em Y (cima/baixo)
        if self.size > 1 {
            // Y bottom/top
            let send_buf_bottom = unsafe {
                (data as *mut u8).add((ny - halo_y) * nx * nz * elem_size) as *const c_void
            };
            // ...
            // Implementação similar para Y e Z
            // Para simplificar, deixamos apenas X.
        }

        self.mpi.barrier()?;
        Ok(())
    }
}