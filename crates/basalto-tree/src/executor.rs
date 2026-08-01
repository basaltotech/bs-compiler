// crates/basalto-tree/src/executor.rs
use anyhow::{Result, anyhow};
use std::ffi::c_void;
use std::sync::Arc;
use tokio::sync::mpsc;
use serde_json::Value;

use basalto_target_nvidia::NvidiaRuntime;
use basalto_common::hardware::GpuIdentity;

// --- Telemetria de energia ---
use energy_telemetry::reader::{AutoEnergyReader, EnergyReader};
use energy_telemetry::correlator::Correlator;

// --- OpenTelemetry (opcional, ativado por feature) ---
#[cfg(feature = "otel")]
use opentelemetry::{global, trace::{Span, Tracer, Status, StatusCode}};
#[cfg(feature = "otel")]
use opentelemetry::trace::TraceContextExt;

// --------------------------------------------------------------------------
// Mensagem para notificar o SiliconForge JIT (background)
// --------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct KernelExecutionReport {
    pub kernel_hash: String,
    pub duration_micros: u64,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u32,
    pub gpu_identity: GpuIdentity,
}

// --------------------------------------------------------------------------
// Executor principal
// --------------------------------------------------------------------------
pub struct Executor {
    runtime: NvidiaRuntime,
    report_sender: Option<mpsc::Sender<KernelExecutionReport>>,
    correlator: Arc<Correlator>, // para registrar consumo de energia
    energy_reader: AutoEnergyReader,
}

impl Executor {
    /// Cria um novo executor, inicializando o runtime CUDA e a telemetria.
    pub fn new(
        report_sender: Option<mpsc::Sender<KernelExecutionReport>>,
        correlator: Arc<Correlator>,
    ) -> Result<Self> {
        let runtime = NvidiaRuntime::new()
            .map_err(|e| anyhow!("Falha ao inicializar CUDA: {}", e))?;

        // Detecta automaticamente a melhor fonte de telemetria (IPMI/Redfish/NVML)
        let energy_reader = AutoEnergyReader::auto_detect();

        Ok(Self {
            runtime,
            report_sender,
            correlator,
            energy_reader,
        })
    }

    /// Lança um kernel na GPU com medição de energia e correlação COUN.
    ///
    /// # Parâmetros
    /// - `ptx_binary`: bytes do PTX compilado.
    /// - `function_name`: nome da função kernel (ex: "basalto_kernel").
    /// - `grid_dim`: (x, y, z) – número de blocos.
    /// - `block_dim`: (x, y, z) – número de threads por bloco.
    /// - `shared_mem_bytes`: memória compartilhada dinâmica (bytes).
    /// - `params`: vetor de ponteiros para os argumentos.
    /// - `kernel_hash`: chave BLAKE3 (para correlação).
    /// - `job_id`: identificador do job (vindo do scheduler) – para auditoria.
    ///
    /// # Retorno
    /// - `Ok(())` se o kernel executou sem erros.
    /// - A função registra o consumo de energia e envia relatório ao JIT.
    pub fn launch_kernel(
        &self,
        ptx_binary: &[u8],
        function_name: &str,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &[*const c_void],
        kernel_hash: Option<String>,
        job_id: Option<String>,
    ) -> Result<()> {
        let start = std::time::Instant::now();

        // --- Medição de energia (antes) ---
        let energy_start_joules = self.energy_reader.read_energy_joules()
            .unwrap_or(0.0);
        let power_start_watts = self.energy_reader.read_power_watts()
            .unwrap_or(0.0);

        // --- Executa o kernel ---
        self.runtime.launch(
            ptx_binary,
            function_name,
            grid_dim,
            block_dim,
            shared_mem_bytes,
            params,
        ).map_err(|e| anyhow!("Erro ao lançar kernel: {}", e))?;

        // --- Medição de energia (depois) ---
        let energy_end_joules = self.energy_reader.read_energy_joules()
            .unwrap_or(0.0);
        let power_end_watts = self.energy_reader.read_power_watts()
            .unwrap_or(0.0);

        let elapsed = start.elapsed().as_micros() as u64;

        // --- Correlação COUN: energia consumida neste kernel ---
        let delta_joules = energy_end_joules - energy_start_joules;
        let delta_kwh = delta_joules / 3_600_000.0; // J → kWh

        if let Some(hash) = &kernel_hash {
            // Registra a correlação no banco de auditoria
            self.correlator.record(
                hash,
                job_id.as_deref().unwrap_or("unknown"),
                &self.energy_reader.get_node_id(),
                delta_kwh,
                elapsed,
                grid_dim,
                block_dim,
                shared_mem_bytes,
            )?;
        }

        // --- Opcional: exportar para OpenTelemetry (métricas) ---
        #[cfg(feature = "otel")]
        {
            let tracer = global::tracer("basalto");
            let mut span = tracer.start("kernel_execution");
            span.set_attribute("kernel_hash".to_string(), kernel_hash.clone().unwrap_or_default());
            span.set_attribute("duration_us".to_string(), elapsed as i64);
            span.set_attribute("delta_kwh".to_string(), delta_kwh);
            span.set_attribute("power_start_w".to_string(), power_start_watts as i64);
            span.set_attribute("power_end_w".to_string(), power_end_watts as i64);
            span.set_attribute("grid_x".to_string(), grid_dim.0 as i64);
            span.set_attribute("block_x".to_string(), block_dim.0 as i64);
            span.set_attribute("shared_mem_bytes".to_string(), shared_mem_bytes as i64);
            span.end();
        }

        // --- Envia relatório para o SiliconForge JIT (se configurado) ---
        if let Some(sender) = &self.report_sender {
            if let Some(hash) = kernel_hash {
                let gpu = GpuIdentity::from_system()
                    .unwrap_or_else(|_| GpuIdentity::default());
                let report = KernelExecutionReport {
                    kernel_hash: hash,
                    duration_micros: elapsed,
                    grid: grid_dim,
                    block: block_dim,
                    shared_mem_bytes,
                    gpu_identity: gpu,
                };
                // Envio não-bloqueante – se o canal estiver cheio, descarta
                let _ = sender.try_send(report);
            }
        }

        Ok(())
    }
}

// --------------------------------------------------------------------------
// Função auxiliar para construir parâmetros do kernel
// --------------------------------------------------------------------------
pub fn build_kernel_params(
    device_ptr_a: *mut c_void,
    device_ptr_b: *mut c_void,
    n: i32,
) -> Vec<*const c_void> {
    vec![
        device_ptr_a as *const c_void,
        device_ptr_b as *const c_void,
        &n as *const i32 as *const c_void,
    ]
}

// --------------------------------------------------------------------------
// Função de alto nível usada pelo interceptor
// --------------------------------------------------------------------------
pub fn execute_flir_kernel(
    ptx_bytes: &[u8],
    function_name: &str,
    flir_params: &Value,
    input_device_ptrs: &[*const c_void],
    output_device_ptrs: &[*const c_void],
    n: i32,
    kernel_hash: Option<String>,
    job_id: Option<String>,
    sender: Option<mpsc::Sender<KernelExecutionReport>>,
    correlator: Arc<Correlator>,
) -> Result<()> {
    // 1. Ler parâmetros do FLIR
    let tile_size = flir_params["tile_size"].as_i64().unwrap_or(128) as u32;
    let shared_mem_bytes = flir_params["shared_mem_bytes"].as_u64().unwrap_or(0) as u32;

    // 2. Calcular grid/block
    let block_x = tile_size.min(1024);
    let grid_x = ((n as u64 + block_x as u64 - 1) / block_x as u64) as u32;
    let grid = (grid_x, 1u32, 1u32);
    let block = (block_x, 1u32, 1u32);

    // 3. Montar parâmetros do kernel: (x, y, N)
    let mut params: Vec<*const c_void> = Vec::with_capacity(3);
    if !input_device_ptrs.is_empty() {
        params.push(input_device_ptrs[0]);
    } else {
        return Err(anyhow!("Nenhum ponteiro de entrada fornecido"));
    }
    if !output_device_ptrs.is_empty() {
        params.push(output_device_ptrs[0]);
    } else {
        return Err(anyhow!("Nenhum ponteiro de saída fornecido"));
    }
    let n_value = n;
    params.push(&n_value as *const i32 as *const c_void);

    // 4. Criar executor e lançar
    let executor = Executor::new(sender, correlator)?;
    executor.launch_kernel(
        ptx_bytes,
        function_name,
        grid,
        block,
        shared_mem_bytes,
        &params,
        kernel_hash,
        job_id,
    )
}

// --------------------------------------------------------------------------
// GpuIdentity::default() para fallback (caso a detecção falhe)
// --------------------------------------------------------------------------
impl Default for GpuIdentity {
    fn default() -> Self {
        Self {
            vendor: "unknown".to_string(),
            arch: "unknown".to_string(),
            driver_version: "unknown".to_string(),
            node_id: "unknown-node".to_string(),
            capabilities: None,
        }
    }
}