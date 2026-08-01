use anyhow::{Result, anyhow};
use std::ffi::c_void;
use tokio::sync::mpsc;
use serde_json::Value;
use basalto_target_nvidia::NvidiaRuntime;
use basalto_common::hardware::GpuIdentity;

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
}

impl Executor {
    pub fn new(report_sender: Option<mpsc::Sender<KernelExecutionReport>>) -> Result<Self> {
        let runtime = NvidiaRuntime::new()
            .map_err(|e| anyhow!("Falha ao inicializar CUDA: {}", e))?;
        Ok(Self { runtime, report_sender })
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
    ) -> Result<()> {
        let start = std::time::Instant::now();
        self.runtime.launch(ptx_binary, function_name, grid_dim, block_dim, shared_mem_bytes, params)
            .map_err(|e| anyhow!("Erro ao lançar kernel: {}", e))?;
        let elapsed = start.elapsed().as_micros() as u64;
        if let Some(sender) = &self.report_sender {
            if let Some(hash) = kernel_hash {
                let gpu = GpuIdentity::from_system().unwrap_or_default();
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
    n: i32,
    kernel_hash: Option<String>,
    sender: Option<mpsc::Sender<KernelExecutionReport>>,
) -> Result<()> {
    let tile_size = flir_params["tile_size"].as_i64().unwrap_or(128) as u32;
    let shared_mem_bytes = flir_params["shared_mem_bytes"].as_u64().unwrap_or(0) as u32;

    let block_x = tile_size.min(1024);
    let grid_x = ((n as u64 + block_x as u64 - 1) / block_x as u64) as u32;
    let grid = (grid_x, 1u32, 1u32);
    let block = (block_x, 1u32, 1u32);

    let mut params: Vec<*const c_void> = Vec::new();
    if !input_device_ptrs.is_empty() { params.push(input_device_ptrs[0]); }
    else { return Err(anyhow!("Sem ponteiro de entrada")); }
    if !output_device_ptrs.is_empty() { params.push(output_device_ptrs[0]); }
    else { return Err(anyhow!("Sem ponteiro de saída")); }
    let n_value = n;
    params.push(&n_value as *const i32 as *const c_void);

    let executor = Executor::new(sender)?;
    executor.launch_kernel(ptx_bytes, function_name, grid, block, shared_mem_bytes, &params, kernel_hash)
}