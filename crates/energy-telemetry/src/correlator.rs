use anyhow::Result;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

// ⭐ Agregador global em memória (por job)
static JOB_AGGREGATOR: Mutex<HashMap<String, JobAggregate>> = Mutex::new(HashMap::new());

/// Representa um registro de energia de uma execução de kernel.
#[derive(Debug, Clone)]
pub struct EnergyRecord {
    pub kernel_hash: u64,
    pub start: u64,                // timestamp em ms
    pub end: u64,                  // timestamp em ms
    pub kwh_delta: f64,            // economia de energia (Baseline - Real)
    pub slurm_job_id: String,      // SLURM_JOB_ID
    pub slurm_step_id: String,     // SLURM_STEP_ID (opcional)
    pub partition: String,         // SLURM_JOB_PARTITION (ex: "gpu-nvidia", "gpu-amd")
    pub operation_type: String,    // "RTM_STENCIL", "FWI", "MATMUL", etc.
    pub vendor: String,            // "nvidia", "amd", "intel"
    pub arch: String,              // "sm_90", "gfx1100"
}

/// Agregado por job (soma de todos os deltas de energia do job)
#[derive(Debug, Clone)]
pub struct JobAggregate {
    pub slurm_job_id: String,
    pub partition: String,
    pub total_kwh_saved: f64,
    pub total_executions: u64,
    pub first_start: u64,
    pub last_end: u64,
    pub vendor: String,
    pub arch: String,
}

/// Cria um registro a partir dos dados da execução.
/// Lê automaticamente as variáveis de ambiente do Slurm.
pub fn create_record(
    kernel_hash: u64,
    kwh_delta: f64,
    operation_type: Option<String>,
    vendor: String,
    arch: String,
) -> EnergyRecord {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let job_id = std::env::var("SLURM_JOB_ID").unwrap_or_else(|_| "unknown".to_string());
    let step_id = std::env::var("SLURM_STEP_ID").unwrap_or_else(|_| "0".to_string());
    let partition = std::env::var("SLURM_JOB_PARTITION").unwrap_or_else(|_| "default".to_string());
    let op_type = operation_type.unwrap_or_else(|| "generic".to_string());

    EnergyRecord {
        kernel_hash,
        start: now - 1,  // aproximação (idealmente passado pelo executor)
        end: now,
        kwh_delta,
        slurm_job_id: job_id,
        slurm_step_id: step_id,
        partition,
        operation_type: op_type,
        vendor,
        arch,
    }
}

/// Adiciona um registro ao agregador global.
pub fn record_delta(record: EnergyRecord) -> Result<()> {
    let mut agg = JOB_AGGREGATOR.lock().unwrap();
    let entry = agg.entry(record.slurm_job_id.clone()).or_insert(JobAggregate {
        slurm_job_id: record.slurm_job_id.clone(),
        partition: record.partition.clone(),
        total_kwh_saved: 0.0,
        total_executions: 0,
        first_start: record.start,
        last_end: record.end,
        vendor: record.vendor.clone(),
        arch: record.arch.clone(),
    });

    entry.total_kwh_saved += record.kwh_delta;
    entry.total_executions += 1;
    if record.start < entry.first_start {
        entry.first_start = record.start;
    }
    if record.end > entry.last_end {
        entry.last_end = record.end;
    }

    // Se atingir 1000 execuções no mesmo job, flush parcial (evita uso excessivo de memória)
    if entry.total_executions % 1000 == 0 {
        drop(agg); // libera o lock antes de flush
        let _ = flush_job_aggregate(&record.slurm_job_id);
    }

    Ok(())
}

/// Envia o agregado do job para o Redis/API de faturamento.
pub fn flush_job_aggregate(job_id: &str) -> Result<()> {
    let mut agg = JOB_AGGREGATOR.lock().unwrap();
    if let Some(entry) = agg.remove(job_id) {
        // Aqui você pode enviar para Redis, API REST, ou escrever em um arquivo.
        // Para HPC, o mais comum é enviar para um Redis centralizado com TTL longo.
        let payload = serde_json::to_string(&entry)?;
        
        // Exemplo: enviar para Redis (se disponível)
        if let Ok(redis_url) = std::env::var("BASALTO_REDIS_URL") {
            let client = redis::Client::open(redis_url)?;
            let mut conn = client.get_connection()?;
            let key = format!("basalto:energy:job:{}", job_id);
            let _: redis::RedisResult<()> = conn.set_ex(key, payload, 86400 * 30); // 30 dias
        } else {
            // Fallback: escreve em arquivo local (para debug em supercomputadores)
            std::fs::write(
                format!("/tmp/basalto_energy_{}.json", job_id),
                serde_json::to_string_pretty(&entry)?,
            )?;
        }
        return Ok(());
    }
    Ok(())
}

/// Função chamada pelo executor no final do job (capturada via signal ou no fim do script Python).
/// Em HPC, normalmente o job tem um hook de saída (SLURM epilog).
pub fn flush_all() -> Result<()> {
    let keys: Vec<String> = {
        let agg = JOB_AGGREGATOR.lock().unwrap();
        agg.keys().cloned().collect()
    };
    for job_id in keys {
        flush_job_aggregate(&job_id)?;
    }
    Ok(())
}