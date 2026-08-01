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
    shape: &[usize],
    kernel_hash: Option<String>,
    sender: Option<mpsc::Sender<KernelExecutionReport>>,
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
        _ => return Err(anyhow!("Dimensão {} não suportada", dims)),
    };

    // Construir parâmetros: [x, y, Nx, Ny] para 2D
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

    let executor = Executor::new(sender)?;
    executor.launch_kernel(
        ptx_bytes,
        function_name,
        grid,
        block,
        shared_mem_bytes,
        &params,
        kernel_hash,
    )
}