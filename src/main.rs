//! CLI for bitnet-oxidized: demo, infer, info, bench, chat, serve, profile, quantize.

use anyhow::Result;
use bitnet_oxidized::{
    create_demo_model,
    model::{from_pretrained, ModelRegistry},
    validation::{validate_model, validate_model_from_path},
    InferenceEngine, TextGenerator,
};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::info;

#[derive(Parser)]
#[command(name = "bitnet-oxidized")]
#[command(about = "1-bit LLM inference in Rust")]
struct Cli {
    /// Run model validation before the chosen operation (or with Validate command).
    #[arg(long, global = true)]
    validate: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run with a demo model (random small model)
    Demo,

    /// Run inference: --model <name or path> --prompt <text> (e.g. demo or path/to/model.gguf)
    Infer {
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "50")]
        max_tokens: usize,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
    },

    /// Show model information
    Info {
        #[arg(long)]
        model: String,
    },

    /// Run benchmarks
    Bench {
        #[arg(long)]
        model: String,
    },

    /// Interactive chat mode
    Chat {
        #[arg(short, long)]
        model: String,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        #[arg(long, default_value = "50")]
        top_k: usize,
        #[arg(long, default_value = "1.2")]
        repetition_penalty: f32,
        #[arg(long, default_value = "0.5")]
        frequency_penalty: f32,
        #[arg(long, default_value = "0.3")]
        presence_penalty: f32,
        #[arg(long)]
        system_prompt: Option<String>,
        #[arg(long)]
        debug: bool,
        /// Greedy mode: temperature≈0, top_k=1, strong repetition penalty
        #[arg(long)]
        greedy: bool,
    },

    /// Server mode (HTTP API)
    Serve {
        #[arg(short, long)]
        model: String,
        #[arg(short, long, default_value = "8080")]
        port: u16,
        #[arg(long, default_value = "4")]
        batch_size: usize,
    },

    /// Profile model performance
    Profile {
        #[arg(short, long)]
        model: String,
        #[arg(long, default_value = "100")]
        iterations: usize,
    },

    /// Quantize FP32 model to ternary
    Quantize {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },

    /// Test tokenizer: encode/decode and vocab check
    TestTokenizer {
        #[arg(short, long)]
        tokenizer: PathBuf,
        #[arg(short, long)]
        text: String,
    },

    /// Run full validation suite on a model and exit (report to stdout unless --output).
    Validate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Convert model to GGUF (safetensors→GGUF; use Python script for Hugging Face).
    Convert {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Output ternary (I2_S) weights
        #[arg(long, default_value = "true")]
        format_i2s: bool,
    },

    /// Repair GGUF: re-save with correct alignment and v3.
    Repair {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },

    /// Inspect GGUF file: metadata and tensor list.
    Inspect {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        verbose: bool,
    },

    /// List registered models (demo + BITNET_REGISTRY / BITNET_MODEL_*)
    Models,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?),
        )
        .init();

    let cli = Cli::parse();

    if cli.validate {
        run_validate_if_requested(&cli)?;
    }

    match cli.command {
        Commands::Validate { model, output } => {
            run_validate(&model, output.as_deref())?;
            return Ok(());
        }
        Commands::Demo => run_demo()?,
        Commands::Infer {
            model,
            prompt,
            max_tokens,
            tokenizer,
        } => run_infer(&model, &prompt, max_tokens, tokenizer.as_deref())?,
        Commands::Info { model } => run_info(&model)?,
        Commands::Bench { model } => run_bench(&model)?,
        Commands::Chat {
            model,
            tokenizer,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            system_prompt,
            debug,
            greedy,
        } => {
            let config = if greedy {
                eprintln!(
                    "Greedy mode enabled (temperature≈0, top_k=1, strong repetition penalty)"
                );
                bitnet_oxidized::GenerationConfig {
                    max_tokens: 100,
                    temperature: 0.01,
                    top_p: 1.0,
                    top_k: Some(1),
                    repetition_penalty: 2.0,
                    frequency_penalty: 1.0,
                    presence_penalty: 0.5,
                    eos_token_id: Some(2),
                }
            } else {
                bitnet_oxidized::GenerationConfig {
                    max_tokens: 100,
                    temperature,
                    top_p,
                    top_k: Some(top_k),
                    repetition_penalty,
                    frequency_penalty,
                    presence_penalty,
                    eos_token_id: Some(2),
                }
            };
            run_chat(
                &model,
                tokenizer.as_deref(),
                config,
                system_prompt.as_deref(),
                debug,
            )?
        }
        Commands::Serve {
            model,
            port,
            batch_size,
        } => run_serve(&model, port, batch_size)?,
        Commands::Profile { model, iterations } => run_profile(&model, iterations)?,
        Commands::Quantize { input, output } => run_quantize(&input, &output)?,
        Commands::TestTokenizer { tokenizer, text } => run_test_tokenizer(&tokenizer, &text)?,
        Commands::Convert {
            input,
            output,
            format_i2s,
        } => run_convert(&input, &output, format_i2s)?,
        Commands::Repair { model, output } => run_repair(&model, &output)?,
        Commands::Inspect { model, verbose } => run_inspect(&model, verbose)?,
        Commands::Models => run_models()?,
    }
    Ok(())
}

fn get_registry() -> ModelRegistry {
    let mut r = ModelRegistry::default_registry();
    r.load_from_env();
    r
}

fn load_model_from_name_or_path(name_or_path: &str) -> Result<bitnet_oxidized::BitNetModel> {
    from_pretrained(name_or_path, &get_registry()).map_err(Into::into)
}

fn run_models() -> Result<()> {
    let reg = get_registry();
    let mut names = reg.names();
    names.sort();
    println!("Registered models:");
    for n in &names {
        if let Some(e) = reg.get(n) {
            let desc = e.description.as_deref().unwrap_or("").trim();
            let path_str = e
                .path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "(in-memory)".to_string());
            if desc.is_empty() {
                println!("  {}  {}", n, path_str);
            } else {
                println!("  {}  {}  # {}", n, path_str, desc);
            }
        }
    }
    println!("\nUse --model <name> or --model <path/to/model.gguf> with infer, chat, serve, etc.");
    Ok(())
}

fn run_validate_if_requested(cli: &Cli) -> Result<()> {
    match &cli.command {
        Commands::Validate { .. } => return Ok(()),
        Commands::Demo => {
            let model = create_demo_model();
            let report = validate_model(&model)?;
            if !report.passed {
                eprintln!(
                    "Validation failed (demo model): {:?}",
                    report.gguf_load.errors
                );
                std::process::exit(1);
            }
            info!("Pre-run validation (demo) passed.");
        }
        Commands::Infer { model, .. }
        | Commands::Info { model, .. }
        | Commands::Bench { model, .. }
        | Commands::Chat { model, .. }
        | Commands::Serve { model, .. }
        | Commands::Profile { model, .. } => {
            if Path::new(model).exists() && Path::new(model).is_file() {
                let report = validate_model_from_path(Path::new(model))?;
                if !report.passed {
                    eprintln!("Validation failed: {:?}", report.gguf_load.errors);
                    std::process::exit(1);
                }
                info!("Pre-run validation passed.");
            }
        }
        _ => {}
    }
    Ok(())
}

fn run_validate(model_path: &std::path::Path, output: Option<&std::path::Path>) -> Result<()> {
    let report = validate_model_from_path(model_path)?;
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(p) = output {
        std::fs::write(p, &json)?;
        println!("Validation report written to {}", p.display());
    } else {
        println!("{}", json);
    }
    if !report.passed {
        std::process::exit(1);
    }
    Ok(())
}

fn run_convert(input: &std::path::Path, output: &std::path::Path, format_i2s: bool) -> Result<()> {
    bitnet_oxidized::convert_safetensors_to_gguf(input, output, format_i2s)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("Converted {} -> {}", input.display(), output.display());
    Ok(())
}

fn run_repair(model_path: &std::path::Path, output: &std::path::Path) -> Result<()> {
    bitnet_oxidized::repair_gguf(model_path, output).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("Repaired {} -> {}", model_path.display(), output.display());
    Ok(())
}

fn run_inspect(model_path: &std::path::Path, verbose: bool) -> Result<()> {
    let info = bitnet_oxidized::inspect_gguf(model_path).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("GGUF version: {}", info.version);
    println!("Alignment OK: {}", info.alignment_ok);
    if !info.alignment_errors.is_empty() {
        for e in &info.alignment_errors {
            println!("  alignment: {}", e);
        }
    }
    println!("\nMetadata ({} keys):", info.metadata.len());
    let mut keys: Vec<_> = info.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = info.metadata.get(k).unwrap();
        println!("  {} = {}", k, v);
    }
    println!("\nTensors ({}):", info.tensors.len());
    for t in &info.tensors {
        if verbose {
            println!(
                "  {}  dims={:?}  type={}  offset={}  n_elements={}",
                t.name, t.dimensions, t.type_name, t.offset, t.n_elements
            );
        } else {
            println!("  {}  {:?}  {}", t.name, t.dimensions, t.type_name);
        }
    }
    Ok(())
}

fn run_demo() -> Result<()> {
    info!("Creating demo model...");
    let model = create_demo_model();
    let engine = InferenceEngine::new(model.clone());
    let prompt_ids = vec![0usize, 1, 2];
    info!("Running forward pass on {} tokens...", prompt_ids.len());
    let t0 = Instant::now();
    let logits = engine.forward(&prompt_ids)?;
    let elapsed = t0.elapsed();
    info!("Forward pass: {:?}, logits len = {}", elapsed, logits.len());
    let best = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    info!("Argmax logit index: {}", best);

    info!("Generating 10 tokens (greedy)...");
    let gen = TextGenerator::new(model);
    let ids = gen.generate_greedy(&prompt_ids, prompt_ids.len() + 10)?;
    info!(
        "Generated {} token IDs: {:?}",
        ids.len(),
        &ids[..ids.len().min(15)]
    );
    Ok(())
}

fn run_infer(
    model_name_or_path: &str,
    prompt: &str,
    max_tokens: usize,
    tokenizer_path: Option<&std::path::Path>,
) -> Result<()> {
    info!("Loading model '{}'...", model_name_or_path);
    let model = load_model_from_name_or_path(model_name_or_path)?;
    let gen = TextGenerator::new(model);
    let (prompt_ids, output_text) = if let Some(tok_path) = tokenizer_path {
        let tok = bitnet_oxidized::BitNetTokenizer::from_file(tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
        let ids = tok
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
        let decode = move |ids: &[usize]| {
            tok.decode(ids).unwrap_or_else(|_| {
                ids.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
        };
        (ids, Some(decode))
    } else {
        (simple_tokenizer(prompt).into_iter().collect(), None)
    };
    info!("Prompt token count: {}", prompt_ids.len());
    let ids = gen.generate_greedy(&prompt_ids, prompt_ids.len() + max_tokens)?;
    if let Some(decode) = output_text {
        let text = decode(&ids[prompt_ids.len()..]);
        println!("Generated: {}", text.trim());
    } else {
        let text: String = ids.iter().map(|&i| format!("{} ", i)).collect();
        println!("Generated: {}", text.trim());
    }
    Ok(())
}

fn run_info(model_name_or_path: &str) -> Result<()> {
    let model = load_model_from_name_or_path(model_name_or_path)?;
    println!("vocab_size: {}", model.vocab_size());
    println!("hidden_size: {}", model.hidden_size());
    println!("num_layers: {}", model.num_layers());
    Ok(())
}

fn run_bench(model_name_or_path: &str) -> Result<()> {
    info!("Loading model '{}'...", model_name_or_path);
    let model = load_model_from_name_or_path(model_name_or_path)?;
    let engine = InferenceEngine::new(model);
    let input = vec![0usize];
    let warmup = 5;
    let iters = 20;
    for _ in 0..warmup {
        let _ = engine.forward(&input)?;
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = engine.forward(&input)?;
    }
    let elapsed = t0.elapsed();
    let per_pass = elapsed.as_secs_f64() / iters as f64 * 1000.0;
    println!("Forward pass: {:.2} ms ({} iters)", per_pass, iters);
    Ok(())
}

fn run_chat(
    model_name_or_path: &str,
    tokenizer_path: Option<&std::path::Path>,
    config: bitnet_oxidized::GenerationConfig,
    _system_prompt: Option<&str>,
    debug: bool,
) -> Result<()> {
    info!("Loading model '{}'...", model_name_or_path);
    let model = load_model_from_name_or_path(model_name_or_path)?;
    let tokenizer = tokenizer_path
        .map(bitnet_oxidized::BitNetTokenizer::from_file)
        .transpose()
        .map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;

    if debug {
        let engine = InferenceEngine::new(model.clone());
        let test_ids = vec![1usize, 2, 3];
        println!(
            "\n--- DEBUG: Forward pass on test token IDs {:?} ---",
            test_ids
        );
        let logits = engine.forward(&test_ids)?;
        let min_l = logits.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean_l = logits.iter().sum::<f32>() / logits.len() as f32;
        println!(
            "  Logits: len={}, min={:.4}, max={:.4}, mean={:.4}",
            logits.len(),
            min_l,
            max_l,
            mean_l
        );
        if logits.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            println!("  WARNING: Logits contain NaN or Inf!");
        }
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("  Top 5 predictions:");
        for (idx, (tid, logit)) in indexed.iter().take(5).enumerate() {
            let token_text = tokenizer
                .as_ref()
                .and_then(|t| t.decode(&[*tid]).ok())
                .unwrap_or_else(|| format!("{}", tid));
            println!(
                "    {}: token_id={} logit={:.2} text={:?}",
                idx + 1,
                tid,
                logit,
                token_text
            );
        }
        println!("--- END DEBUG ---\n");
    }

    let gen = TextGenerator::new(model);
    println!("Chat. Type 'quit' to exit.");
    println!("  temperature={}, top_p={}, top_k={:?}, repetition_penalty={}, frequency_penalty={}, presence_penalty={}",
        config.temperature, config.top_p, config.top_k, config.repetition_penalty, config.frequency_penalty, config.presence_penalty);
    loop {
        print!("> ");
        use std::io::Write;
        std::io::stdout().flush()?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err()
            || line.trim().eq_ignore_ascii_case("quit")
        {
            break;
        }
        let ids = if let Some(ref tok) = tokenizer {
            // LLaMA-style chat format: <s>[INST] user message [/INST]  (model then generates assistant reply)
            let formatted = format!("<s>[INST] {} [/INST] ", line.trim());
            tok.encode(&formatted).unwrap_or_default()
        } else {
            simple_tokenizer(line.trim()).into_iter().collect()
        };
        if ids.is_empty() {
            continue;
        }
        let out = gen.generate_with_config(&ids, &config, debug)?;
        let eos_id = config.eos_token_id.unwrap_or(2);
        let new_ids: Vec<usize> = out
            .iter()
            .skip(ids.len())
            .take_while(|&&id| id != eos_id)
            .copied()
            .collect();
        let text = if let Some(ref tok) = tokenizer {
            tok.decode(&new_ids).unwrap_or_else(|_| {
                new_ids
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
        } else {
            new_ids
                .iter()
                .map(|&i| format!("{} ", i))
                .collect::<String>()
        };
        println!("{}", text.trim());
    }
    Ok(())
}

fn run_serve(model_name_or_path: &str, port: u16, batch_size: usize) -> Result<()> {
    let config = bitnet_oxidized::server::ServerConfig {
        max_batch_size: batch_size,
        max_queue_size: 64,
        timeout_seconds: 30,
    };
    let telemetry = Some(std::sync::Arc::new(bitnet_oxidized::Telemetry::new()));
    let rt = tokio::runtime::Runtime::new().map_err(|e| anyhow::anyhow!("tokio runtime: {}", e))?;
    rt.block_on(bitnet_oxidized::server::run_server_with_registry(
        model_name_or_path,
        &get_registry(),
        None::<&std::path::Path>,
        port,
        config,
        telemetry,
    ))
    .map_err(|e| anyhow::anyhow!("server: {}", e))
}

fn run_profile(model_name_or_path: &str, iterations: usize) -> Result<()> {
    info!("Loading model '{}'...", model_name_or_path);
    let model = load_model_from_name_or_path(model_name_or_path)?;
    let engine = InferenceEngine::new(model);
    let input = vec![0usize];
    let warmup = 5;
    for _ in 0..warmup {
        let _ = engine.forward(&input)?;
    }
    let t0 = Instant::now();
    for _ in 0..iterations {
        let _ = engine.forward(&input)?;
    }
    let elapsed = t0.elapsed();
    let per_pass_ms = elapsed.as_secs_f64() / iterations as f64 * 1000.0;
    println!(
        "Profile: {} iters, {:.3} ms/forward",
        iterations, per_pass_ms
    );
    Ok(())
}

fn run_quantize(input: &std::path::Path, output: &std::path::Path) -> Result<()> {
    use bitnet_oxidized::model::gguf;
    let model = gguf::load_gguf(input).map_err(|e| anyhow::anyhow!("load GGUF: {:?}", e))?;
    gguf::save_gguf(&model, output).map_err(|e| anyhow::anyhow!("save GGUF: {:?}", e))?;
    println!("Saved quantized model to {}", output.display());
    Ok(())
}

fn run_test_tokenizer(tokenizer_path: &std::path::Path, text: &str) -> Result<()> {
    let tok = bitnet_oxidized::BitNetTokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
    let ids = tok
        .encode(text)
        .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
    let decoded = tok
        .decode(&ids)
        .map_err(|e| anyhow::anyhow!("decode: {}", e))?;
    println!("Input: {}", text);
    println!("Token IDs: {:?}", ids);
    println!("Decoded: {}", decoded);
    println!("\nVocab check:");
    for id in &ids {
        if *id >= 32000 {
            println!(
                "  WARNING: Token ID {} may be out of range (vocab often 32000-32002)",
                id
            );
        }
    }
    Ok(())
}

/// Naive tokenizer: split on whitespace and map to token IDs (hash % vocab).
fn simple_tokenizer(prompt: &str) -> Vec<usize> {
    const VOCAB: usize = 256;
    prompt
        .split_whitespace()
        .map(|s| {
            let h = s.bytes().fold(0u64, |a, b| a.wrapping_add(b as u64));
            (h as usize) % VOCAB
        })
        .collect()
}
