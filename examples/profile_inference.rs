//! Profile inference: per-layer timing, FLOPS estimate, optional Chrome trace export.
//!
//! Usage:
//!   cargo run --example profile_inference -- (demo model)
//!   cargo run --example profile_inference -- --model path/to/model.gguf
//!   cargo run --example profile_inference -- --model path/to/model.gguf --trace trace.json

use anyhow::Result;
use bitnet_oxidized::{
    create_demo_model,
    model::BitNetModel,
    profiling::{InferenceProfiler, ProfilerReport},
};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "profile_inference")]
struct Args {
    /// Path to GGUF model file. If omitted, uses in-memory demo model.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Output Chrome trace JSON (for chrome://tracing). Default: no file.
    #[arg(long)]
    trace: Option<PathBuf>,

    /// Number of tokens in the dummy input sequence.
    #[arg(long, default_value = "8")]
    seq_len: usize,

    /// Number of warmup forward passes before profiling.
    #[arg(long, default_value = "2")]
    warmup: usize,

    /// Number of profiled forward passes (report is from the last one).
    #[arg(long, default_value = "1")]
    runs: usize,
}

fn load_model(args: &Args) -> Result<BitNetModel> {
    if let Some(path) = &args.model {
        if !path.exists() {
            anyhow::bail!("Model file not found: {}", path.display());
        }
        Ok(bitnet_oxidized::model::load_gguf(path)?)
    } else {
        Ok(create_demo_model())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model = load_model(&args)?;
    let profiler = InferenceProfiler::new(model);

    let vocab_size = profiler.model().vocab_size();
    let input_ids: Vec<usize> = (0..args.seq_len).map(|i| i % vocab_size).collect();

    for _ in 0..args.warmup {
        let _ = profiler.forward_profiled(&input_ids)?;
    }

    let mut last_report: Option<ProfilerReport> = None;
    for _ in 0..args.runs {
        let (_, report) = profiler.forward_profiled(&input_ids)?;
        last_report = Some(report);
    }

    let report = last_report.expect("at least one run");
    println!("=== Profiler report ===");
    println!("Tokens: {}", report.tokens_processed);
    println!("Total:   {:.3} ms", report.total_ms);
    println!(
        "FLOPS:   {} ({:.2} GFLOP)",
        report.total_flops_estimate,
        report.total_flops_estimate as f64 / 1e9
    );
    println!("\nPer-layer:");
    for t in &report.layer_timings {
        println!(
            "  {}  {:.3} ms  ({} FLOP)",
            t.name, t.duration_ms, t.flops_estimate
        );
    }

    if let Some(trace_path) = &args.trace {
        let events = InferenceProfiler::report_to_chrome_trace(&report, 1, 1);
        let json = serde_json::to_string_pretty(&events)?;
        std::fs::write(trace_path, &json)?;
        println!("\nChrome trace written to {}", trace_path.display());
    }

    Ok(())
}
