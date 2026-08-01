use anyhow::Result;
use std::time::Instant;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::LazyLock;

// 📊 Banco de dados de Heurísticas Global (In-Memory)
// Mapeia o formato do tensor para o histórico de tempo de execução na GPU.
#[derive(Default)]
pub struct HeuristicRegistry {
    // Chave: Representação em string do formato do shape (ex: "1x3x224x224")
    // Valor: Vetor com os últimos tempos de execução em microssegundos
    pub execution_history: HashMap<String, Vec<u128>>,
    // Guarda o melhor layout de memória descoberto para este formato até agora
    pub best_layout_strategy: HashMap<String, String>,
}

static REGISTRY: LazyLock<Arc<Mutex<HeuristicRegistry>>> = LazyLock::new(|| {
    Arc::new(Mutex::new(HeuristicRegistry::default()))
});

/// Registra de forma assíncrona os dados de execução e recalibra as decisões do compilador.
pub async fn record_execution(binary: &[u8], shapes: &[Vec<usize>]) -> Result<()> {
    if binary.is_empty() || shapes.is_empty() {
        return Ok(()); // Nada para registrar
    }

    // 1. Cria uma assinatura textual única para o formato atual dos tensores
    // Exemplo: [[1, 3, 224, 224]] vira "1_3_224_224"
    let shape_signature = serialize_shapes_signature(shapes);

    // 2. Coleta de Métricas de Hardware (Simulação de leitura de Eventos CUDA / NVML)
    // Em produção, você captura o tempo usando CUDA Events (`cudaEventElapsedTime`) 
    // para medir o tempo PURO da GPU, isolado do overhead da CPU.
    let gpu_execution_time_us = capture_gpu_hardware_time(binary);

    // 3. Atualiza o registro global sem travar a thread principal por muito tempo
    // Como a função é async, o tokio/async-std pode gerenciar esse lock em background.
    let registry_clone = Arc::clone(&REGISTRY);
    
    // Despachamos a computação pesada de análise para não engargalar o runtime
    tokio::task::spawn_blocking(move || {
        let mut registry = registry_clone.lock().unwrap();
        
        let history = registry.execution_history.entry(shape_signature.clone()).or_insert_with(Vec::new);
        history.push(gpu_execution_time_us);

        // Mantém apenas os últimos 100 registros para não estourar a memória RAM
        if history.len() > 100 {
            history.remove(0);
        }

        // 4. RECALIBRAÇÃO DE HEURÍSTICA (O "Cérebro" do Autotuner)
        // Se o tempo médio de execução atual começou a subir ou degradar, 
        // o compilador decide mudar a estratégia de geração de código para a próxima vez.
        let average_time: u128 = history.iter().sum::<u128>() / history.len() as u128;
        
        if average_time > 5000 { // Exemplo: se passou de 5 milissegundos
            // Alerta o compilador Basalto que para este shape específico, 
            // a estratégia de Padding ou vetorização precisa mudar.
            registry.best_layout_strategy.insert(
                shape_signature, 
                "FORCE_TILE_ALIGNMENT_32".to_string()
            );
        }
    }).await?;

    Ok(())
}

fn serialize_shapes_signature(shapes: &[Vec<usize>]) -> String {
    shapes.iter()
        .map(|shape| shape.iter().map(|dim| dim.to_string()).collect::<Vec<_>>().join("_"))
        .collect::<Vec<_>>()
        .join("-")
}

fn capture_gpu_hardware_time(_binary: &[u8]) -> u128 {
    // Placeholder: Na prática, lê os registradores de telemetria da GPU
    // Retorna o tempo de execução em microssegundos
    4200 
}
