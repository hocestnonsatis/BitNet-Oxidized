//! Run full validation suite on a BitNet model and export report to JSON.
//!
//! Usage:
//!   cargo run --example validate_model -- --model path/to/model.gguf
//!   cargo run --example validate_model -- --model path/to/model.gguf --output report.json
//!   cargo run --example validate_model -- (uses demo model, no file)

use anyhow::Result;
use bitnet_oxidized::{
    create_demo_model,
    validation::{validate_model, validate_model_from_path, ValidationReport},
};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "validate_model")]
struct Args {
    /// Path to GGUF model file. If omitted, uses in-memory demo model.
    #[arg(long)]
    model: Option<PathBuf>,

    /// Output JSON report path. Default: print to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let report: ValidationReport = if let Some(path) = &args.model {
        if !path.exists() {
            eprintln!("Model file not found: {}", path.display());
            std::process::exit(1);
        }
        validate_model_from_path(path)?
    } else {
        let model = create_demo_model();
        validate_model(&model)?
    };

    let json = serde_json::to_string_pretty(&report)?;

    if let Some(out) = &args.output {
        std::fs::write(out, &json)?;
        println!("Validation report written to {}", out.display());
    } else {
        println!("{}", json);
    }

    if !report.passed {
        eprintln!("Validation failed.");
        std::process::exit(1);
    }

    println!("Validation passed.");
    Ok(())
}
