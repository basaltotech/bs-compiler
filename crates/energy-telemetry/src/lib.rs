// PROPRIETÁRIO: Telemetria de energia (COUN)
pub mod reader;
pub mod correlator;

pub use correlator::{
    EnergyRecord,
    JobAggregate,
    create_record,
    record_delta,
    flush_job_aggregate,
    flush_all,
};