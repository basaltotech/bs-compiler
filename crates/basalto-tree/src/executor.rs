use anyhow::{anyhow, Result};
use std::ffi::c_void;
use std::sync::Arc;
use tokio::sync::mpsc;
use serde_json::Value;
use basalto_target_nvidia::NvidiaRuntime;
use basalto_common::hardware::GpuIdentity;
use energy_telemetry::correlator::Correlator;
use energy_telemetry::comparator::TemporalComparator;
use siliconforge_jit::profiler::KernelExecutionRecord;

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
    halo_exchanger: Option<Arc<basalto_communication::halo_exchange::HaloExchanger>>,
}

impl Executor {
    pub fn new(
        report_sender: Option<mpsc::Sender<KernelExecutionReport>>,
        profiler_sender: Option<mpsc::Sender<KernelExecutionRecord>>,
        correlator: Arc<Correlator>,
        comparator: Arc<TemporalComparator>,
        halo_exchanger: Option<Arc<basalto_communication::halo_exchange::HaloExchanger>>,
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
        })
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
        job_id: Option<&str>,
        device_ptr_x: *mut c_void,
        device_ptr_y: *mut c_void,
    ) -> Result<()> {
        // Troca de halos entre GPUs/nós (antes do kernel)
        if let Some(exchanger) = &self.halo_exchanger {
            let elem_size = if dtype == "f32" || dtype == "f16" || dtype == "bf16" { 4 } else { 8 };
            let rank = exchanger.get_rank();
            let size = exchanger.get_size();
            eprintln!(
                "[Executor] Trocando halos (rank={}/{}) para shape={:?}",
                rank, size, shape
            );
            exchanger.exchange_halo_3d(
                device_ptr_x,
                shape[0],
                shape[1],
                shape[2],
                1,  // halo_x
                1,  // halo_y
                1,  // halo_z
                elem_size,
                None, // stream
            )?;
        }

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

        let gpu = GpuIdentity::from_system().unwrap_or_default();

        if let Some(hash) = kernel_hash {
            let job_id_str = job_id.unwrap_or("unknown");
            let node_id = &gpu.node_id;

            self.correlator.record(&hash, job_id_str, node_id, 0.0, elapsed);
            self.comparator.record_execution(&hash, op, dtype, shape, 0);

            if let Some(prev_hash) = self.comparator.get_previous_execution(op, dtype, shape, &hash) {
                if let Some((delta_kwh, delta_percent, delta_duration)) =
                    self.comparator.compute_delta(&hash, &prev_hash)
                {
                    eprintln!(
                        "[4D] Delta: {} kWh ({:.2}%), {} us",
                        delta_kwh, delta_percent * 100.0, delta_duration
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
                    node_id: Some(node_id.clone()),
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
}

pub fn execute_flir_kernel(
    ptx_bytes: &[u8],
    function_name: &str,
    flir_params: &Value,
    input_device_ptrs: &[*const c_void],
    output_device_ptrs: &[*const c_void],
    shape: &[usize],
    strides: &[isize],
    kernel_hash: Option<String>,
    sender: Option<mpsc::Sender<KernelExecutionReport>>,
    profiler_sender: Option<mpsc::Sender<KernelExecutionRecord>>,
    correlator: Arc<Correlator>,
    comparator: Arc<TemporalComparator>,
    op: &str,
    dtype: &str,
    job_id: Option<&str>,
    halo_exchanger: Option<Arc<basalto_communication::halo_exchange::HaloExchanger>>,
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
        job_id,
        device_ptr_x,
        device_ptr_y,
    )
}