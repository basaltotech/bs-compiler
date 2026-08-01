use anyhow::{anyhow, Result};
use std::ffi::c_void;
use std::sync::Arc;
use tokio::sync::mpsc;
use serde_json::Value;

use basalto_target_nvidia::NvidiaRuntime;
use basalto_target_nvidia::blas::{CublasRuntime, CUBLAS_OP_N, CUBLAS_OP_T};
use basalto_common::hardware::GpuIdentity;
use energy_telemetry::reader::{EnergyReader, AutoEnergyReader};
use energy_telemetry::correlator::Correlator;
use energy_telemetry::comparator::TemporalComparator;
use siliconforge_jit::profiler::KernelExecutionRecord;
use basalto_communication::HaloExchanger;

#[cfg(feature = "cutlass")]
use basalto_gemm_jit::fused_kernel::execute_fused_gemm;
#[cfg(feature = "cutlass")]
use basalto_gemm_jit::cutlass::FusedOp;

#[derive(Debug, Clone)]
pub struct KernelExecutionReport {
    pub kernel_hash: String,
    pub duration_micros: u64,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u32,
    pub gpu_identity: GpuIdentity,
}

pub struct Executor {
    runtime: NvidiaRuntime,
    report_sender: Option<mpsc::Sender<KernelExecutionReport>>,
    profiler_sender: Option<mpsc::Sender<KernelExecutionRecord>>,
    correlator: Arc<Correlator>,
    comparator: Arc<TemporalComparator>,
    halo_exchanger: Option<Arc<HaloExchanger>>,
    energy_reader: AutoEnergyReader,
}

impl Executor {
    pub fn new(
        report_sender: Option<mpsc::Sender<KernelExecutionReport>>,
        profiler_sender: Option<mpsc::Sender<KernelExecutionRecord>>,
        correlator: Arc<Correlator>,
        comparator: Arc<TemporalComparator>,
        halo_exchanger: Option<Arc<HaloExchanger>>,
    ) -> Result<Self> {
        let runtime = NvidiaRuntime::new()
            .map_err(|e| anyhow!("Falha ao inicializar CUDA: {}", e))?;
        Ok(Self {
            runtime,
            report_sender,
            profiler_sender,
            correlator,
            comparator,
            halo_exchanger,
            energy_reader: AutoEnergyReader::auto_detect(),
        })
    }

    fn sample_energy_mj(&self) -> Result<u64> {
        if let Some(nvml) = self.energy_reader.get_nvml() {
            unsafe { nvml.get_total_energy_mj() }
        } else {
            let joules = self.energy_reader.read_energy_joules()?;
            Ok((joules * 1000.0) as u64)
        }
    }

    pub fn launch_kernel(
        &self,
        ptx_binary: &[u8],
        function_name: &str,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &[*const c_void],
        kernel_hash: Option<String>,
        op: &str,
        dtype: &str,
        shape: &[usize],
        strides: &[isize],
        radius: usize,
        job_id: Option<&str>,
        device_ptr_x: *mut c_void,
        device_ptr_y: *mut c_void,
    ) -> Result<()> {
        if let Some(exchanger) = &self.halo_exchanger {
            let elem_size = if dtype == "f32" || dtype == "f16" || dtype == "bf16" { 4 } else { 8 };
            let dims = shape.len();
            exchanger.exchange_halo_3d(
                device_ptr_x,
                shape[0],
                if dims >= 2 { shape[1] } else { 1 },
                if dims >= 3 { shape[2] } else { 1 },
                radius,
                if dims >= 2 { radius } else { 0 },
                if dims >= 3 { radius } else { 0 },
                elem_size,
                None,
            )?;
        }

        let energy_before_mj = self.sample_energy_mj().unwrap_or(0);
        let start = std::time::Instant::now();

        self.runtime
            .launch(
                ptx_binary,
                function_name,
                grid_dim,
                block_dim,
                shared_mem_bytes,
                params,
            )
            .map_err(|e| anyhow!("Erro ao lançar kernel: {}", e))?;

        let elapsed = start.elapsed().as_micros() as u64;

        let energy_after_mj = self.sample_energy_mj().unwrap_or(0);
        let delta_joules = (energy_after_mj - energy_before_mj) as f64 / 1000.0;
        let delta_kwh = delta_joules / 3_600_000.0;

        let gpu = GpuIdentity::from_system().unwrap_or_default();
        if let Some(hash) = kernel_hash {
            let job_id_str = job_id.unwrap_or("unknown");
            self.correlator.record(&hash, job_id_str, &gpu.node_id, delta_kwh, elapsed);
            self.comparator.record_execution(&hash, op, dtype, shape, 0);

            if let Some(prev_hash) = self.comparator.get_previous_execution(op, dtype, shape, &hash) {
                if let Some((d_kwh, d_pct, d_dur)) =
                    self.comparator.compute_delta(&hash, &prev_hash)
                {
                    eprintln!(
                        "[4D] Delta: {} kWh ({:.2}%), {} us",
                        d_kwh, d_pct * 100.0, d_dur
                    );
                }
            }

            if let Some(sender) = &self.profiler_sender {
                let record = KernelExecutionRecord {
                    kernel_hash: hash.clone(),
                    duration_us: elapsed,
                    grid: grid_dim,
                    block: block_dim,
                    shared_mem_bytes,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    job_id: Some(job_id_str.to_string()),
                    node_id: Some(gpu.node_id.clone()),
                    gpu_vendor: gpu.vendor.clone(),
                    gpu_arch: gpu.arch.clone(),
                    driver_version: gpu.driver_version.clone(),
                };
                let _ = sender.try_send(record);
            }
        }

        if let Some(sender) = &self.report_sender {
            if let Some(hash) = kernel_hash {
                let report = KernelExecutionReport {
                    kernel_hash: hash,
                    duration_micros: elapsed,
                    grid: grid_dim,
                    block: block_dim,
                    shared_mem_bytes,
                    gpu_identity: gpu,
                };
                let _ = sender.try_send(report);
            }
        }

        Ok(())
    }

    pub fn validate_kernel(
        &self,
        op: &str,
        dtype: &str,
        shape: &[usize],
        gpu_result_ptr: *mut c_void,
        a_ptr: Option<*mut c_void>,
        b_ptr: Option<*mut c_void>,
        input_ptr: Option<*mut c_void>,
        m: Option<usize>,
        n: Option<usize>,
        k: Option<usize>,
        radius: Option<usize>,
        coeffs: Option<Vec<f64>>,
    ) -> Result<()> {
        let elem_size = if dtype == "f32" { 4 } else { 8 };
        let atol = 1e-5;
        let rtol = 1e-5;

        let result_len = match op {
            "matmul" => {
                let m = m.ok_or_else(|| anyhow!("m não fornecido"))?;
                let n = n.ok_or_else(|| anyhow!("n não fornecido"))?;
                let batch = shape.first().unwrap_or(&1);
                batch * m * n
            }
            "stencil_1d" | "stencil_2d" | "stencil_3d" => shape.iter().product(),
            _ => return Err(anyhow!("Validação não implementada para {}", op)),
        };

        let mut gpu_result_cpu = vec![0u8; result_len * elem_size];
        unsafe {
            let cuda = basalto_communication::CudaRuntime::new()?;
            cuda.memcpy(
                gpu_result_cpu.as_mut_ptr() as *mut c_void,
                gpu_result_ptr,
                gpu_result_cpu.len(),
                basalto_communication::cuda::CUDA_MEMCPY_DEVICE_TO_HOST,
            )?;
        }

        let cpu_result: Vec<f64> = match op {
            "stencil_1d" => {
                vec![0.0; result_len]
            }
            _ => vec![0.0; result_len],
        };

        let gpu_vals: Vec<f64> = gpu_result_cpu
            .chunks_exact(elem_size)
            .map(|chunk| {
                if dtype == "f32" {
                    f32::from_ne_bytes(chunk.try_into().unwrap()) as f64
                } else {
                    f64::from_ne_bytes(chunk.try_into().unwrap())
                }
            })
            .collect();

        for (i, (gv, cv)) in gpu_vals.iter().zip(cpu_result.iter()).enumerate() {
            let diff = (gv - cv).abs();
            let tolerance = atol + rtol * cv.abs();
            if diff > tolerance {
                return Err(anyhow!(
                    "Validação falhou no índice {}: GPU={:.6e}, CPU={:.6e}, diff={:.6e}, tol={:.6e}",
                    i, gv, cv, diff, tolerance
                ));
            }
        }

        eprintln!("[Validator] Validação numérica passou ({} elementos)", result_len);
        Ok(())
    }

    pub fn execute_cublas(
        &self,
        a_ptr: *mut c_void,
        b_ptr: *mut c_void,
        c_ptr: *mut c_void,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        batch: usize,
        dtype: &str,
        kernel_hash: Option<String>,
        job_id: Option<&str>,
    ) -> Result<()> {
        let cublas = CublasRuntime::new()
            .map_err(|e| anyhow!("Falha ao carregar cuBLAS: {}", e))?;
        let handle = cublas.create_handle()?;

        let start = std::time::Instant::now();

        let op_a = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let op_b = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };

        let a_stride = m * k;
        let b_stride = k * n;
        let c_stride = m * n;
        let elem_size = if dtype == "f32" { 4 } else { 8 };

        unsafe {
            match dtype {
                "f32" => {
                    let alpha: f32 = 1.0;
                    let beta: f32 = 0.0;
                    for i in 0..batch {
                        let a_offset = (a_ptr as *mut u8).add(i * a_stride * elem_size);
                        let b_offset = (b_ptr as *mut u8).add(i * b_stride * elem_size);
                        let c_offset = (c_ptr as *mut u8).add(i * c_stride * elem_size);
                        let status = (cublas.cublas_sgemm)(
                            handle,
                            op_a, op_b,
                            m as i32, n as i32, k as i32,
                            &alpha,
                            a_offset as *const c_void, m as i32,
                            b_offset as *const c_void, k as i32,
                            &beta,
                            c_offset as *mut c_void, m as i32,
                        );
                        if status != 0 {
                            return Err(anyhow!("cublasSgemm falhou no batch {} com status {}", i, status));
                        }
                    }
                }
                "f64" => {
                    let alpha: f64 = 1.0;
                    let beta: f64 = 0.0;
                    for i in 0..batch {
                        let a_offset = (a_ptr as *mut u8).add(i * a_stride * elem_size);
                        let b_offset = (b_ptr as *mut u8).add(i * b_stride * elem_size);
                        let c_offset = (c_ptr as *mut u8).add(i * c_stride * elem_size);
                        let status = (cublas.cublas_dgemm)(
                            handle,
                            op_a, op_b,
                            m as i32, n as i32, k as i32,
                            &alpha,
                            a_offset as *const c_void, m as i32,
                            b_offset as *const c_void, k as i32,
                            &beta,
                            c_offset as *mut c_void, m as i32,
                        );
                        if status != 0 {
                            return Err(anyhow!("cublasDgemm falhou no batch {} com status {}", i, status));
                        }
                    }
                }
                _ => return Err(anyhow!("Tipo de dado não suportado para cuBLAS: {}", dtype)),
            }
        }

        let elapsed = start.elapsed().as_micros() as u64;
        let _ = cublas.destroy_handle(handle);

        if let Some(hash) = kernel_hash {
            let gpu = GpuIdentity::from_system().unwrap_or_default();
            let job_id_str = job_id.unwrap_or("unknown");
            self.correlator.record(&hash, job_id_str, &gpu.node_id, 0.0, elapsed);
            self.comparator.record_execution(&hash, "matmul", &[m, k, n], 0);
        }

        Ok(())
    }

    #[cfg(feature = "cutlass")]
    pub fn execute_fused_gemm(
        &self,
        a_ptr: *mut c_void,
        b_ptr: *mut c_void,
        c_ptr: *mut c_void,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        batch: usize,
        dtype: &str,
        fused_op: FusedOp,
        arch: &str,
        kernel_hash: Option<String>,
        job_id: Option<&str>,
    ) -> Result<()> {
        let start = std::time::Instant::now();

        let ptx = execute_fused_gemm(
            a_ptr, b_ptr, c_ptr,
            m, n, k,
            trans_a, trans_b,
            batch,
            dtype,
            fused_op,
            arch,
        )?;

        let elapsed = start.elapsed().as_micros() as u64;

        if let Some(hash) = kernel_hash {
            let gpu = GpuIdentity::from_system().unwrap_or_default();
            let job_id_str = job_id.unwrap_or("unknown");
            self.correlator.record(&hash, job_id_str, &gpu.node_id, 0.0, elapsed);
            self.comparator.record_execution(&hash, "fused_gemm", &[m, k, n], 0);
        }

        Ok(())
    }
}

pub fn execute_flir_kernel(
    ptx_bytes: &[u8],
    function_name: &str,
    flir_params: &Value,
    input_device_ptrs: &[*const c_void],
    output_device_ptrs: &[*const c_void],
    shape: &[usize],
    strides: &[isize],
    radius: usize,
    kernel_hash: Option<String>,
    sender: Option<mpsc::Sender<KernelExecutionReport>>,
    profiler_sender: Option<mpsc::Sender<KernelExecutionRecord>>,
    correlator: Arc<Correlator>,
    comparator: Arc<TemporalComparator>,
    op: &str,
    dtype: &str,
    job_id: Option<&str>,
    halo_exchanger: Option<Arc<HaloExchanger>>,
    device_ptr_x: *mut c_void,
    device_ptr_y: *mut c_void,
) -> Result<()> {
    let dims = shape.len();
    let tile_x = flir_params["tile_x"].as_i64().unwrap_or(128) as u32;
    let tile_y = flir_params["tile_y"].as_i64().unwrap_or(1) as u32;
    let shared_mem_bytes = flir_params["shared_mem_bytes"].as_u64().unwrap_or(0) as u32;

    let (grid, block) = match dims {
        1 => {
            let n = shape[0] as u32;
            let block_x = tile_x.min(1024);
            let grid_x = (n + block_x - 1) / block_x;
            ((grid_x, 1, 1), (block_x, 1, 1))
        }
        2 => {
            let n_x = shape[0] as u32;
            let n_y = shape[1] as u32;
            let block_x = tile_x.min(1024);
            let block_y = tile_y.min(1024);
            let grid_x = (n_x + block_x - 1) / block_x;
            let grid_y = (n_y + block_y - 1) / block_y;
            ((grid_x, grid_y, 1), (block_x, block_y, 1))
        }
        3 => {
            let n_x = shape[0] as u32;
            let n_y = shape[1] as u32;
            let block_x = tile_x.min(1024);
            let block_y = tile_y.min(1024);
            let grid_x = (n_x + block_x - 1) / block_x;
            let grid_y = (n_y + block_y - 1) / block_y;
            ((grid_x, grid_y, 1), (block_x, block_y, 1))
        }
        _ => return Err(anyhow!("Dimensão {} não suportada", dims)),
    };

    let mut params: Vec<*const c_void> = Vec::new();
    if input_device_ptrs.is_empty() || output_device_ptrs.is_empty() {
        return Err(anyhow!("Ponteiros de entrada/saída não fornecidos"));
    }
    params.push(input_device_ptrs[0]);
    params.push(output_device_ptrs[0]);

    if dims >= 1 {
        let nx = shape[0] as i32;
        params.push(&nx as *const i32 as *const c_void);
    }
    if dims >= 2 {
        let ny = shape[1] as i32;
        params.push(&ny as *const i32 as *const c_void);
    }
    if dims >= 3 {
        let nz = shape[2] as i32;
        params.push(&nz as *const i32 as *const c_void);
    }

    if dims >= 1 {
        let sx = strides[0] as i32;
        params.push(&sx as *const i32 as *const c_void);
    } else {
        let sx = 1;
        params.push(&sx as *const i32 as *const c_void);
    }
    if dims >= 2 {
        let sy = strides[1] as i32;
        params.push(&sy as *const i32 as *const c_void);
    } else {
        let sy = 1;
        params.push(&sy as *const i32 as *const c_void);
    }
    if dims >= 3 {
        let sz = strides[2] as i32;
        params.push(&sz as *const i32 as *const c_void);
    } else {
        let sz = 1;
        params.push(&sz as *const i32 as *const c_void);
    }

    let executor = Executor::new(
        sender,
        profiler_sender,
        correlator,
        comparator,
        halo_exchanger,
    )?;

    executor.launch_kernel(
        ptx_bytes,
        function_name,
        grid,
        block,
        shared_mem_bytes,
        &params,
        kernel_hash,
        op,
        dtype,
        shape,
        strides,
        radius,
        job_id,
        device_ptr_x,
        device_ptr_y,
    )
}

pub fn execute_cublas_kernel(
    a_ptr: *mut c_void,
    b_ptr: *mut c_void,
    c_ptr: *mut c_void,
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
    batch: usize,
    dtype: &str,
    kernel_hash: Option<String>,
    sender: Option<mpsc::Sender<KernelExecutionReport>>,
    profiler_sender: Option<mpsc::Sender<KernelExecutionRecord>>,
    correlator: Arc<Correlator>,
    comparator: Arc<TemporalComparator>,
    job_id: Option<&str>,
) -> Result<()> {
    let executor = Executor::new(sender, profiler_sender, correlator, comparator, None)?;

    let energy_before_mj = executor.sample_energy_mj().unwrap_or(0);
    let start = std::time::Instant::now();

    executor.execute_cublas(
        a_ptr, b_ptr, c_ptr,
        m, n, k,
        trans_a, trans_b,
        batch,
        dtype,
        kernel_hash.clone(),
        job_id,
    )?;

    let elapsed = start.elapsed().as_micros() as u64;
    let energy_after_mj = executor.sample_energy_mj().unwrap_or(0);
    let delta_joules = (energy_after_mj - energy_before_mj) as f64 / 1000.0;
    let delta_kwh = delta_joules / 3_600_000.0;

    if let Some(hash) = kernel_hash {
        let gpu = GpuIdentity::from_system().unwrap_or_default();
        let job_id_str = job_id.unwrap_or("unknown");
        executor.correlator.record(&hash, job_id_str, &gpu.node_id, delta_kwh, elapsed);
        executor.comparator.record_execution(&hash, "matmul", &[m, k, n], 0);
    }

    Ok(())
}

#[cfg(feature = "cutlass")]
pub fn execute_fused_gemm_kernel(
    a_ptr: *mut c_void,
    b_ptr: *mut c_void,
    c_ptr: *mut c_void,
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
    batch: usize,
    dtype: &str,
    fused_op: FusedOp,
    arch: &str,
    kernel_hash: Option<String>,
    sender: Option<mpsc::Sender<KernelExecutionReport>>,
    profiler_sender: Option<mpsc::Sender<KernelExecutionRecord>>,
    correlator: Arc<Correlator>,
    comparator: Arc<TemporalComparator>,
    job_id: Option<&str>,
) -> Result<()> {
    let executor = Executor::new(sender, profiler_sender, correlator, comparator, None)?;

    let energy_before_mj = executor.sample_energy_mj().unwrap_or(0);
    let start = std::time::Instant::now();

    executor.execute_fused_gemm(
        a_ptr, b_ptr, c_ptr,
        m, n, k,
        trans_a, trans_b,
        batch,
        dtype,
        fused_op,
        arch,
        kernel_hash.clone(),
        job_id,
    )?;

    let elapsed = start.elapsed().as_micros() as u64;
    let energy_after_mj = executor.sample_energy_mj().unwrap_or(0);
    let delta_joules = (energy_after_mj - energy_before_mj) as f64 / 1000.0;
    let delta_kwh = delta_joules / 3_600_000.0;

    if let Some(hash) = kernel_hash {
        let gpu = GpuIdentity::from_system().unwrap_or_default();
        let job_id_str = job_id.unwrap_or("unknown");
        executor.correlator.record(&hash, job_id_str, &gpu.node_id, delta_kwh, elapsed);
        executor.comparator.record_execution(&hash, "fused_gemm", &[m, k, n], 0);
    }

    Ok(())
}