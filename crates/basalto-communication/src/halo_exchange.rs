use anyhow::{anyhow, Result};
use std::ffi::c_void;
use std::sync::Arc;
use crate::{MpiRuntime, NcclRuntime, CudaRuntime};

pub struct HaloExchanger {
    mpi: Arc<MpiRuntime>,
    nccl: Option<Arc<NcclRuntime>>,
    cuda: Arc<CudaRuntime>,
    rank: i32,
    size: i32,
    gpu_aware_mpi: bool,
}

impl HaloExchanger {
    pub fn new(
        mpi: Arc<MpiRuntime>,
        nccl: Option<Arc<NcclRuntime>>,
        cuda: Arc<CudaRuntime>,
    ) -> Result<Self> {
        let rank = mpi.rank()?;
        let size = mpi.size()?;
        let gpu_aware_mpi = Self::detect_gpu_aware_mpi();
        Ok(Self {
            mpi,
            nccl,
            cuda,
            rank,
            size,
            gpu_aware_mpi,
        })
    }

    pub fn get_rank(&self) -> i32 {
        self.rank
    }

    pub fn get_size(&self) -> i32 {
        self.size
    }

    fn detect_gpu_aware_mpi() -> bool {
        std::env::var("MV2_USE_CUDA").map(|s| s == "1").unwrap_or(false) ||
        std::env::var("OMPI_MCA_mpi_cuda_support").map(|s| s == "1").unwrap_or(false) ||
        std::env::var("MPICH_GPU_SUPPORT_ENABLED").map(|s| s == "1").unwrap_or(false)
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
        stream: Option<*mut c_void>,
    ) -> Result<()> {
        if self.size <= 1 {
            return Ok(());
        }

        let data_u8 = data as *mut u8;
        let left_rank = (self.rank - 1 + self.size) % self.size;
        let right_rank = (self.rank + 1) % self.size;
        let bottom_rank = (self.rank - 1 + self.size) % self.size;
        let top_rank = (self.rank + 1) % self.size;
        let front_rank = (self.rank - 1 + self.size) % self.size;
        let back_rank = (self.rank + 1) % self.size;

        // Troca em X
        if halo_x > 0 {
            let halo_x_bytes = halo_x * ny * nz * elem_size;
            let send_right = self.copy_region_to_cpu(
                data_u8, nx, ny, nz, elem_size,
                nx - halo_x, 0, 0,
                halo_x, ny, nz,
            )?;
            let mut recv_left = vec![0u8; halo_x_bytes];
            self.mpi.sendrecv(
                send_right.as_ptr() as *const c_void,
                send_right.len() as i32,
                right_rank,
                recv_left.as_mut_ptr() as *mut c_void,
                recv_left.len() as i32,
                left_rank,
            )?;
            self.copy_buffer_to_gpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                halo_x, ny, nz,
                &recv_left,
            )?;

            let send_left = self.copy_region_to_cpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                halo_x, ny, nz,
            )?;
            let mut recv_right = vec![0u8; halo_x_bytes];
            self.mpi.sendrecv(
                send_left.as_ptr() as *const c_void,
                send_left.len() as i32,
                left_rank,
                recv_right.as_mut_ptr() as *mut c_void,
                recv_right.len() as i32,
                right_rank,
            )?;
            self.copy_buffer_to_gpu(
                data_u8, nx, ny, nz, elem_size,
                nx - halo_x, 0, 0,
                halo_x, ny, nz,
                &recv_right,
            )?;
        }

        // Troca em Y
        if halo_y > 0 && self.size > 1 {
            let halo_y_bytes = nx * halo_y * nz * elem_size;
            let send_top = self.copy_region_to_cpu(
                data_u8, nx, ny, nz, elem_size,
                0, ny - halo_y, 0,
                nx, halo_y, nz,
            )?;
            let mut recv_bottom = vec![0u8; halo_y_bytes];
            self.mpi.sendrecv(
                send_top.as_ptr() as *const c_void,
                send_top.len() as i32,
                top_rank,
                recv_bottom.as_mut_ptr() as *mut c_void,
                recv_bottom.len() as i32,
                bottom_rank,
            )?;
            self.copy_buffer_to_gpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                nx, halo_y, nz,
                &recv_bottom,
            )?;

            let send_bottom = self.copy_region_to_cpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                nx, halo_y, nz,
            )?;
            let mut recv_top = vec![0u8; halo_y_bytes];
            self.mpi.sendrecv(
                send_bottom.as_ptr() as *const c_void,
                send_bottom.len() as i32,
                bottom_rank,
                recv_top.as_mut_ptr() as *mut c_void,
                recv_top.len() as i32,
                top_rank,
            )?;
            self.copy_buffer_to_gpu(
                data_u8, nx, ny, nz, elem_size,
                0, ny - halo_y, 0,
                nx, halo_y, nz,
                &recv_top,
            )?;
        }

        // Troca em Z
        if halo_z > 0 && self.size > 1 {
            let halo_z_bytes = nx * ny * halo_z * elem_size;
            let send_back = self.copy_region_to_cpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, nz - halo_z,
                nx, ny, halo_z,
            )?;
            let mut recv_front = vec![0u8; halo_z_bytes];
            self.mpi.sendrecv(
                send_back.as_ptr() as *const c_void,
                send_back.len() as i32,
                back_rank,
                recv_front.as_mut_ptr() as *mut c_void,
                recv_front.len() as i32,
                front_rank,
            )?;
            self.copy_buffer_to_gpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                nx, ny, halo_z,
                &recv_front,
            )?;

            let send_front = self.copy_region_to_cpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                nx, ny, halo_z,
            )?;
            let mut recv_back = vec![0u8; halo_z_bytes];
            self.mpi.sendrecv(
                send_front.as_ptr() as *const c_void,
                send_front.len() as i32,
                front_rank,
                recv_back.as_mut_ptr() as *mut c_void,
                recv_back.len() as i32,
                back_rank,
            )?;
            self.copy_buffer_to_gpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, nz - halo_z,
                nx, ny, halo_z,
                &recv_back,
            )?;
        }

        self.mpi.barrier()?;
        Ok(())
    }

    fn copy_region_to_cpu(
        &self,
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
    ) -> Result<Vec<u8>> {
        let total_bytes = len_x * len_y * len_z * elem_size;
        let mut buf = vec![0u8; total_bytes];
        let src_offset = (start_z * ny * nx + start_y * nx + start_x) * elem_size;
        let src_ptr = unsafe { data.add(src_offset) };
        unsafe {
            self.cuda.memcpy(
                buf.as_mut_ptr() as *mut c_void,
                src_ptr as *const c_void,
                total_bytes,
                crate::cuda::CUDA_MEMCPY_DEVICE_TO_HOST,
            )?;
        }
        Ok(buf)
    }

    fn copy_buffer_to_gpu(
        &self,
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
    ) -> Result<()> {
        let total_bytes = len_x * len_y * len_z * elem_size;
        let dst_offset = (start_z * ny * nx + start_y * nx + start_x) * elem_size;
        let dst_ptr = unsafe { data.add(dst_offset) };
        unsafe {
            self.cuda.memcpy(
                dst_ptr as *mut c_void,
                buffer.as_ptr() as *const c_void,
                total_bytes,
                crate::cuda::CUDA_MEMCPY_HOST_TO_DEVICE,
            )?;
        }
        Ok(())
    }
}