use anyhow::{anyhow, Result};
use std::ffi::c_void;
use std::ptr;
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

    // ----------------------------------------------------------------------
    // Funções auxiliares para copiar regiões do volume 3D para buffers lineares
    // ----------------------------------------------------------------------
    fn copy_region_to_buffer(
        data: *const u8,
        nx: usize,
        ny: usize,
        nz: usize,
        elem_size: usize,
        start_x: usize,
        start_y: usize,
        start_z: usize,
        len_x: usize,
        len_y: usize,
        len_z: usize,
    ) -> Vec<u8> {
        let total_bytes = len_x * len_y * len_z * elem_size;
        let mut buf = vec![0u8; total_bytes];
        let mut idx = 0;

        for z in start_z..start_z + len_z {
            for y in start_y..start_y + len_y {
                let src_offset = (z * ny * nx + y * nx + start_x) * elem_size;
                let src_ptr = unsafe { data.add(src_offset) };
                let dst_ptr = &mut buf[idx];
                unsafe {
                    ptr::copy_nonoverlapping(src_ptr, dst_ptr, len_x * elem_size);
                }
                idx += len_x * elem_size;
            }
        }
        buf
    }

    fn copy_buffer_to_region(
        data: *mut u8,
        nx: usize,
        ny: usize,
        nz: usize,
        elem_size: usize,
        start_x: usize,
        start_y: usize,
        start_z: usize,
        len_x: usize,
        len_y: usize,
        len_z: usize,
        buffer: &[u8],
    ) {
        let mut idx = 0;
        for z in start_z..start_z + len_z {
            for y in start_y..start_y + len_y {
                let dst_offset = (z * ny * nx + y * nx + start_x) * elem_size;
                let dst_ptr = unsafe { data.add(dst_offset) };
                let src_ptr = &buffer[idx];
                unsafe {
                    ptr::copy_nonoverlapping(src_ptr, dst_ptr, len_x * elem_size);
                }
                idx += len_x * elem_size;
            }
        }
    }

    // ----------------------------------------------------------------------
    // Troca de halos 3D completa (X, Y, Z) usando MPI com buffers temporários
    // ----------------------------------------------------------------------
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
        // Se não houver MPI ou for single-node, não faz nada
        if self.size <= 1 {
            return Ok(());
        }

        let data_u8 = data as *mut u8;
        let left_rank = (self.rank - 1 + self.size) % self.size;
        let right_rank = (self.rank + 1) % self.size;

        // ======================================================================
        // 1. TROCA EM X (esquerda / direita)
        // ======================================================================
        // Halo esquerdo: x = [0 .. halo_x-1]
        // Halo direito : x = [nx - halo_x .. nx-1]
        if halo_x > 0 {
            // --- Enviar halo direito para a direita, receber da esquerda ---
            let send_right = Self::copy_region_to_buffer(
                data_u8, nx, ny, nz, elem_size,
                nx - halo_x, 0, 0,
                halo_x, ny, nz,
            );
            let mut recv_left = vec![0u8; halo_x * ny * nz * elem_size];

            self.mpi.sendrecv(
                send_right.as_ptr() as *const c_void,
                send_right.len() as i32,
                right_rank,
                recv_left.as_mut_ptr() as *mut c_void,
                recv_left.len() as i32,
                left_rank,
            )?;

            Self::copy_buffer_to_region(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                halo_x, ny, nz,
                &recv_left,
            );

            // --- Enviar halo esquerdo para a esquerda, receber da direita ---
            let send_left = Self::copy_region_to_buffer(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                halo_x, ny, nz,
            );
            let mut recv_right = vec![0u8; halo_x * ny * nz * elem_size];

            self.mpi.sendrecv(
                send_left.as_ptr() as *const c_void,
                send_left.len() as i32,
                left_rank,
                recv_right.as_mut_ptr() as *mut c_void,
                recv_right.len() as i32,
                right_rank,
            )?;

            Self::copy_buffer_to_region(
                data_u8, nx, ny, nz, elem_size,
                nx - halo_x, 0, 0,
                halo_x, ny, nz,
                &recv_right,
            );
        }

        // ======================================================================
        // 2. TROCA EM Y (baixo / cima)
        // ======================================================================
        // Halo inferior: y = [0 .. halo_y-1]
        // Halo superior : y = [ny - halo_y .. ny-1]
        if halo_y > 0 {
            // Neste caso, o vizinho em Y é o mesmo rank? Em uma decomposição 1D,
            // apenas X é distribuído. Para Y e Z, normalmente o halo é local.
            // Mas implementamos o mecanismo completo por segurança.
            let bottom_rank = (self.rank - 1 + self.size) % self.size;
            let top_rank = (self.rank + 1) % self.size;

            // Enviar halo superior para cima, receber de baixo
            let send_top = Self::copy_region_to_buffer(
                data_u8, nx, ny, nz, elem_size,
                0, ny - halo_y, 0,
                nx, halo_y, nz,
            );
            let mut recv_bottom = vec![0u8; nx * halo_y * nz * elem_size];

            self.mpi.sendrecv(
                send_top.as_ptr() as *const c_void,
                send_top.len() as i32,
                top_rank,
                recv_bottom.as_mut_ptr() as *mut c_void,
                recv_bottom.len() as i32,
                bottom_rank,
            )?;

            Self::copy_buffer_to_region(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                nx, halo_y, nz,
                &recv_bottom,
            );

            // Enviar halo inferior para baixo, receber de cima
            let send_bottom = Self::copy_region_to_buffer(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                nx, halo_y, nz,
            );
            let mut recv_top = vec![0u8; nx * halo_y * nz * elem_size];

            self.mpi.sendrecv(
                send_bottom.as_ptr() as *const c_void,
                send_bottom.len() as i32,
                bottom_rank,
                recv_top.as_mut_ptr() as *mut c_void,
                recv_top.len() as i32,
                top_rank,
            )?;

            Self::copy_buffer_to_region(
                data_u8, nx, ny, nz, elem_size,
                0, ny - halo_y, 0,
                nx, halo_y, nz,
                &recv_top,
            );
        }

        // ======================================================================
        // 3. TROCA EM Z (frente / trás)
        // ======================================================================
        // Halo frontal: z = [0 .. halo_z-1]
        // Halo traseiro: z = [nz - halo_z .. nz-1]
        if halo_z > 0 {
            let front_rank = (self.rank - 1 + self.size) % self.size;
            let back_rank = (self.rank + 1) % self.size;

            // Enviar halo traseiro para trás, receber da frente
            let send_back = Self::copy_region_to_buffer(
                data_u8, nx, ny, nz, elem_size,
                0, 0, nz - halo_z,
                nx, ny, halo_z,
            );
            let mut recv_front = vec![0u8; nx * ny * halo_z * elem_size];

            self.mpi.sendrecv(
                send_back.as_ptr() as *const c_void,
                send_back.len() as i32,
                back_rank,
                recv_front.as_mut_ptr() as *mut c_void,
                recv_front.len() as i32,
                front_rank,
            )?;

            Self::copy_buffer_to_region(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                nx, ny, halo_z,
                &recv_front,
            );

            // Enviar halo frontal para frente, receber de trás
            let send_front = Self::copy_region_to_buffer(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                nx, ny, halo_z,
            );
            let mut recv_back = vec![0u8; nx * ny * halo_z * elem_size];

            self.mpi.sendrecv(
                send_front.as_ptr() as *const c_void,
                send_front.len() as i32,
                front_rank,
                recv_back.as_mut_ptr() as *mut c_void,
                recv_back.len() as i32,
                back_rank,
            )?;

            Self::copy_buffer_to_region(
                data_u8, nx, ny, nz, elem_size,
                0, 0, nz - halo_z,
                nx, ny, halo_z,
                &recv_back,
            );
        }

        // Sincroniza todos os ranks
        self.mpi.barrier()?;

        Ok(())
    }
}