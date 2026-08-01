//! Troca de halos 3D entre GPUs/nós usando MPI (GPU‑Aware) e NCCL.
//!
//! Estratégia:
//!   1. Se NCCL estiver disponível e as GPUs estiverem no mesmo nó,
//!      usa ncclSend/ncclRecv (GPU→GPU, zero-copy).
//!   2. Se MPI for GPU‑Aware (OpenMPI com CUDA), passa ponteiros GPU diretamente.
//!   3. Fallback: staging via cudaMemcpy (GPU→CPU→MPI→CPU→GPU).

use std::ffi::c_void;
use std::sync::Arc;
use anyhow::{anyhow, Result};
use crate::{MpiRuntime, NcclRuntime, CudaRuntime};

pub struct HaloExchanger {
    mpi: Arc<MpiRuntime>,
    nccl: Option<Arc<NcclRuntime>>,
    cuda: Arc<CudaRuntime>,
    rank: i32,
    size: i32,
    use_nccl: bool,
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

        // Detecta se MPI é GPU‑Aware (tentando enviar um ponteiro GPU dummy)
        let gpu_aware_mpi = false; // Em produção, testar com MPI_Probe ou feature flag

        // Decide se usa NCCL (apenas se houver >1 GPU no mesmo nó)
        let use_nccl = nccl.is_some();

        Ok(Self {
            mpi,
            nccl,
            cuda,
            rank,
            size,
            use_nccl,
            gpu_aware_mpi,
        })
    }

    pub fn get_rank(&self) -> i32 { self.rank }
    pub fn get_size(&self) -> i32 { self.size }

    /// Troca halos em X, Y e Z entre os ranks vizinhos.
    ///
    /// # Parâmetros
    /// - `data`: ponteiro para o dado na GPU (device pointer)
    /// - `nx, ny, nz`: dimensões do volume local
    /// - `halo_x, halo_y, halo_z`: largura do halo em cada eixo
    /// - `elem_size`: tamanho de cada elemento (ex: 4 para float32)
    /// - `stream`: CUDA stream para operações assíncronas (opcional)
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

        let left_rank = (self.rank - 1 + self.size) % self.size;
        let right_rank = (self.rank + 1) % self.size;

        // Tamanhos dos halos em bytes
        let halo_x_bytes = halo_x * ny * nz * elem_size;
        let halo_y_bytes = nx * halo_y * nz * elem_size;
        let halo_z_bytes = nx * ny * halo_z * elem_size;

        // Se tivermos NCCL, usamos para transferências GPU→GPU (dentro do mesmo nó)
        if self.use_nccl {
            return self.exchange_halo_nccl(
                data, nx, ny, nz, halo_x, halo_y, halo_z, elem_size, stream,
            );
        }

        // Fallback: MPI com staging via CPU
        self.exchange_halo_mpi_staging(
            data, nx, ny, nz, halo_x, halo_y, halo_z, elem_size,
        )
    }

    /// Troca de halos usando NCCL P2P (GPU→GPU, zero-copy).
    #[allow(unused_variables)]
    fn exchange_halo_nccl(
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
        let nccl = self.nccl.as_ref().ok_or_else(|| anyhow!("NCCL não disponível"))?;
        let left_rank = (self.rank - 1 + self.size) % self.size;
        let right_rank = (self.rank + 1) % self.size;

        // NCCL usa o mesmo tipo para todos os ranks do comunicador.
        // Para simplificar, assumimos que todos os ranks estão no mesmo comunicador.
        let nccl_comm = 0; // Em produção, obter de um communicator global

        // TODO: Implementar troca P2P com ncclSend/ncclRecv
        // Por enquanto, fallback para staging
        self.exchange_halo_mpi_staging(
            data, nx, ny, nz, halo_x, halo_y, halo_z, elem_size,
        )
    }

    /// Troca de halos via MPI com staging (CPU buffers).
    /// Esta é a implementação mais portável e funciona em qualquer cluster.
    fn exchange_halo_mpi_staging(
        &self,
        data: *mut c_void,
        nx: usize,
        ny: usize,
        nz: usize,
        halo_x: usize,
        halo_y: usize,
        halo_z: usize,
        elem_size: usize,
    ) -> Result<()> {
        let data_u8 = data as *mut u8;
        let left_rank = (self.rank - 1 + self.size) % self.size;
        let right_rank = (self.rank + 1) % self.size;

        // Tamanhos dos halos
        let halo_x_bytes = halo_x * ny * nz * elem_size;
        let halo_y_bytes = nx * halo_y * nz * elem_size;
        let halo_z_bytes = nx * ny * halo_z * elem_size;

        // ====================================================================
        // 1. TROCA EM X (esquerda / direita)
        // ====================================================================
        if halo_x > 0 {
            // --- Halo direito → enviar para direita, receber da esquerda ---
            let send_buf = self.copy_region_to_cpu(
                data_u8, nx, ny, nz, elem_size,
                nx - halo_x, 0, 0,
                halo_x, ny, nz,
            )?;
            let mut recv_buf = vec![0u8; halo_x_bytes];

            self.mpi.sendrecv(
                send_buf.as_ptr() as *const c_void,
                send_buf.len() as i32,
                right_rank,
                recv_buf.as_mut_ptr() as *mut c_void,
                recv_buf.len() as i32,
                left_rank,
            )?;

            self.copy_buffer_to_gpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                halo_x, ny, nz,
                &recv_buf,
            )?;

            // --- Halo esquerdo → enviar para esquerda, receber da direita ---
            let send_buf = self.copy_region_to_cpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                halo_x, ny, nz,
            )?;
            let mut recv_buf = vec![0u8; halo_x_bytes];

            self.mpi.sendrecv(
                send_buf.as_ptr() as *const c_void,
                send_buf.len() as i32,
                left_rank,
                recv_buf.as_mut_ptr() as *mut c_void,
                recv_buf.len() as i32,
                right_rank,
            )?;

            self.copy_buffer_to_gpu(
                data_u8, nx, ny, nz, elem_size,
                nx - halo_x, 0, 0,
                halo_x, ny, nz,
                &recv_buf,
            )?;
        }

        // ====================================================================
        // 2. TROCA EM Y (baixo / cima)
        // ====================================================================
        if halo_y > 0 && self.size > 1 {
            let bottom_rank = (self.rank - 1 + self.size) % self.size;
            let top_rank = (self.rank + 1) % self.size;

            // Halo superior → enviar para cima, receber de baixo
            let send_buf = self.copy_region_to_cpu(
                data_u8, nx, ny, nz, elem_size,
                0, ny - halo_y, 0,
                nx, halo_y, nz,
            )?;
            let mut recv_buf = vec![0u8; halo_y_bytes];

            self.mpi.sendrecv(
                send_buf.as_ptr() as *const c_void,
                send_buf.len() as i32,
                top_rank,
                recv_buf.as_mut_ptr() as *mut c_void,
                recv_buf.len() as i32,
                bottom_rank,
            )?;

            self.copy_buffer_to_gpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                nx, halo_y, nz,
                &recv_buf,
            )?;

            // Halo inferior → enviar para baixo, receber de cima
            let send_buf = self.copy_region_to_cpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                nx, halo_y, nz,
            )?;
            let mut recv_buf = vec![0u8; halo_y_bytes];

            self.mpi.sendrecv(
                send_buf.as_ptr() as *const c_void,
                send_buf.len() as i32,
                bottom_rank,
                recv_buf.as_mut_ptr() as *mut c_void,
                recv_buf.len() as i32,
                top_rank,
            )?;

            self.copy_buffer_to_gpu(
                data_u8, nx, ny, nz, elem_size,
                0, ny - halo_y, 0,
                nx, halo_y, nz,
                &recv_buf,
            )?;
        }

        // ====================================================================
        // 3. TROCA EM Z (frente / trás)
        // ====================================================================
        if halo_z > 0 && self.size > 1 {
            let front_rank = (self.rank - 1 + self.size) % self.size;
            let back_rank = (self.rank + 1) % self.size;

            // Halo traseiro → enviar para trás, receber da frente
            let send_buf = self.copy_region_to_cpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, nz - halo_z,
                nx, ny, halo_z,
            )?;
            let mut recv_buf = vec![0u8; halo_z_bytes];

            self.mpi.sendrecv(
                send_buf.as_ptr() as *const c_void,
                send_buf.len() as i32,
                back_rank,
                recv_buf.as_mut_ptr() as *mut c_void,
                recv_buf.len() as i32,
                front_rank,
            )?;

            self.copy_buffer_to_gpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                nx, ny, halo_z,
                &recv_buf,
            )?;

            // Halo frontal → enviar para frente, receber de trás
            let send_buf = self.copy_region_to_cpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, 0,
                nx, ny, halo_z,
            )?;
            let mut recv_buf = vec![0u8; halo_z_bytes];

            self.mpi.sendrecv(
                send_buf.as_ptr() as *const c_void,
                send_buf.len() as i32,
                front_rank,
                recv_buf.as_mut_ptr() as *mut c_void,
                recv_buf.len() as i32,
                back_rank,
            )?;

            self.copy_buffer_to_gpu(
                data_u8, nx, ny, nz, elem_size,
                0, 0, nz - halo_z,
                nx, ny, halo_z,
                &recv_buf,
            )?;
        }

        self.mpi.barrier()?;
        Ok(())
    }

    // ========================================================================
    // AUXILIARES: cópia GPU ↔ CPU com cudaMemcpy
    // ========================================================================

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