// crates/basalto-tree/src/executor.rs
use anyhow::{Result, anyhow};
use std::ffi::c_void;
use tokio::sync::mpsc;
use basalto_target_nvidia::NvidiaRuntime;
use basalto_common::hardware::GpuIdentity;
use serde_json::Value;

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
    // Canal para enviar relatórios ao SiliconForge (opcional)
    report_sender: Option<mpsc::Sender<KernelExecutionReport>>,
}

impl Executor {
    /// Cria um novo executor, inicializando o runtime CUDA.
    pub fn new(report_sender: Option<mpsc::Sender<KernelExecutionReport>>) -> Result<Self> {
        let runtime = NvidiaRuntime::new()
            .map_err(|e| anyhow!("Falha ao inicializar CUDA: {}", e))?;
        Ok(Self { runtime, report_sender })
    }

    /// Lança um kernel na GPU.
    ///
    /// # Parâmetros
    /// - `ptx_binary`: bytes do PTX compilado (gerado pelo `compile_to_ptx`).
    /// - `function_name`: nome da função kernel (ex: "basalto_kernel").
    /// - `grid_dim`: (x, y, z) – número de blocos.
    /// - `block_dim`: (x, y, z) – número de threads por bloco.
    /// - `shared_mem_bytes`: quantidade de memória compartilhada dinâmica (em bytes).
    /// - `params`: vetor de ponteiros para os argumentos (cada um `*const c_void`).
    ///
    /// # Retorno
    /// - `Ok(())` se o kernel executou sem erros.
    /// - A função também envia um relatório para o canal do SiliconForge (se configurado).
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

        // Executa o lançamento via runtime NVIDIA
        self.runtime.launch(
            ptx_binary,
            function_name,
            grid_dim,
            block_dim,
            shared_mem_bytes,
            params,
        ).map_err(|e| anyhow!("Erro ao lançar kernel: {}", e))?;

        let elapsed = start.elapsed().as_micros() as u64;

        // Se houver canal, envia relatório para o SiliconForge (não bloqueante)
        if let Some(sender) = &self.report_sender {
            if let Some(hash) = kernel_hash {
                let gpu = GpuIdentity::from_system();
                let report = KernelExecutionReport {
                    kernel_hash: hash,
                    duration_micros: elapsed,
                    grid: grid_dim,
                    block: block_dim,
                    shared_mem_bytes,
                    gpu_identity: gpu,
                };
                // Envio assíncrono – se o canal estiver cheio, descarta o relatório
                let _ = sender.try_send(report);
            }
        }

        Ok(())
    }
}

// --------------------------------------------------------------------------
// Função auxiliar para construir os parâmetros do kernel a partir de
// ponteiros de dispositivos e valores escalares.
// --------------------------------------------------------------------------
pub fn build_kernel_params(
    device_ptr_a: *mut c_void,
    device_ptr_b: *mut c_void,
    n: i32,
) -> Vec<*const c_void> {
    let mut params: Vec<*const c_void> = Vec::new();
    params.push(device_ptr_a as *const c_void);
    params.push(device_ptr_b as *const c_void);
    params.push(&n as *const i32 as *const c_void);
    params
}

// --------------------------------------------------------------------------
// Integração com o interceptor: recebe o PTX e metadados do FLIR,
// calcula grid/block, e dispara o kernel.
// --------------------------------------------------------------------------
pub fn execute_flir_kernel(
    ptx_bytes: &[u8],
    function_name: &str,
    flir_params: &Value,
    input_device_ptrs: &[*const c_void],  // ponteiros para os tensores de entrada
    output_device_ptrs: &[*const c_void], // ponteiro para o tensor de saída
    n: i32,                               // tamanho do array (ex: número de elementos)
    kernel_hash: Option<String>,
    sender: Option<mpsc::Sender<KernelExecutionReport>>,
) -> Result<()> {
    // 1. Ler tile_size e shared_mem_bytes dos parâmetros FLIR
    let tile_size = flir_params["tile_size"].as_i64().unwrap_or(128) as u32;
    let shared_mem_bytes = flir_params["shared_mem_bytes"].as_u64().unwrap_or(0) as u32;

    // 2. Calcular grid e block
    //    blockDim.x = tile_size (mínimo entre tile_size e max_threads_per_block)
    //    gridDim.x = ceil(N / blockDim.x)
    let block_x = tile_size.min(1024); // limite máximo da GPU (ajustável)
    let grid_x = ((n as u64 + block_x as u64 - 1) / block_x as u64) as u32;

    let grid = (grid_x, 1u32, 1u32);
    let block = (block_x, 1u32, 1u32);

    // 3. Montar lista de parâmetros
    //    Ordem: x (entrada), y (saída), N (int)
    let mut params: Vec<*const c_void> = Vec::new();
    // O primeiro ponteiro é o dispositivo de entrada (ex: x)
    if !input_device_ptrs.is_empty() {
        params.push(input_device_ptrs[0]);
    } else {
        return Err(anyhow!("Nenhum ponteiro de entrada fornecido"));
    }
    // O segundo ponteiro é o dispositivo de saída (ex: y)
    if !output_device_ptrs.is_empty() {
        params.push(output_device_ptrs[0]);
    } else {
        return Err(anyhow!("Nenhum ponteiro de saída fornecido"));
    }
    // O terceiro argumento é o inteiro N (passamos por referência, mas é cópia)
    // Precisamos alocar uma variável local para manter o valor vivo durante a chamada.
    let n_value = n;
    params.push(&n_value as *const i32 as *const c_void);

    // 4. Criar executor e lançar
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