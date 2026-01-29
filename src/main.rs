//! CLI for bitnet-oxidized: demo, infer, info, bench, chat, serve, profile, quantize.

use anyhow::Result;
use bitnet_oxidized::{create_demo_model, InferenceEngine, TextGenerator};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

#[derive(Parser)]
#[command(name = "bitnet-oxidized")]
#[command(about = "1-bit LLM inference in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run with a demo model (random small model)
    Demo,

    /// Run inference: --model <path> --prompt <text> [--tokenizer <path> for decoded text]
    Infer {
        #[arg(long)]
        model: PathBuf,
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
        model: PathBuf,
    },

    /// Run benchmarks
    Bench {
        #[arg(long)]
        model: PathBuf,
    },

    /// Interactive chat mode
    Chat {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long, default_value = "0.8")]
        temperature: f32,
        #[arg(long)]
        system_prompt: Option<String>,
    },

    /// Server mode (HTTP API)
    Serve {
        #[arg(short, long)]
        model: PathBuf,
        #[arg(short, long, default_value = "8080")]
        port: u16,
        #[arg(long, default_value = "4")]
        batch_size: usize,
    },

    /// Profile model performance
    Profile {
        #[arg(short, long)]
        model: PathBuf,
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
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
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
            temperature,
            system_prompt,
        } => run_chat(&model, temperature, system_prompt.as_deref())?,
        Commands::Serve {
            model,
            port,
            batch_size,
        } => run_serve(&model, port, batch_size)?,
        Commands::Profile { model, iterations } => run_profile(&model, iterations)?,
        Commands::Quantize { input, output } => run_quantize(&input, &output)?,
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
    model_path: &std::path::Path,
    prompt: &str,
    max_tokens: usize,
    tokenizer_path: Option<&std::path::Path>,
) -> Result<()> {
    info!("Loading model from {:?}...", model_path);
    let model = bitnet_oxidized::model::gguf::load_gguf(model_path)?;
    let gen = TextGenerator::new(model);
    let (prompt_ids, output_text) = if let Some(tok_path) = tokenizer_path {
        let tok = bitnet_oxidized::BitNetTokenizer::from_file(tok_path)
            .map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
        let ids = tok.encode(prompt).map_err(|e| anyhow::anyhow!("encode: {}", e))?;
        let decode = move |ids: &[usize]| {
            tok.decode(ids).unwrap_or_else(|_| ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "))
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

fn run_info(model_path: &std::path::Path) -> Result<()> {
    let model = bitnet_oxidized::model::gguf::load_gguf(model_path)?;
    println!("vocab_size: {}", model.vocab_size());
    println!("hidden_size: {}", model.hidden_size());
    println!("num_layers: {}", model.num_layers());
    Ok(())
}

fn run_bench(model_path: &std::path::Path) -> Result<()> {
    info!("Loading model from {:?}...", model_path);
    let model = bitnet_oxidized::model::gguf::load_gguf(model_path)?;
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
    model_path: &std::path::Path,
    temperature: f32,
    _system_prompt: Option<&str>,
) -> Result<()> {
    info!("Loading model from {:?}...", model_path);
    let model = if model_path.exists() {
        bitnet_oxidized::model::gguf::load_gguf(model_path)?
    } else {
        info!("Model file not found, using demo model");
        create_demo_model()
    };
    let gen = TextGenerator::new(model);
    println!(
        "Chat (demo). Type 'quit' to exit. Temperature: {}",
        temperature
    );
    let mut prompt_ids = vec![0usize];
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
        let ids = simple_tokenizer(line.trim());
        if ids.is_empty() {
            continue;
        }
        let max_len = prompt_ids.len() + ids.len() + 20;
        let out = gen.generate_top_p(&ids, max_len, 0.9, temperature)?;
        let text: String = out
            .iter()
            .skip(ids.len())
            .map(|&i| format!("{} ", i))
            .take(20)
            .collect();
        println!("{}", text.trim());
        prompt_ids = out;
    }
    Ok(())
}

fn run_serve(model_path: &std::path::Path, port: u16, batch_size: usize) -> Result<()> {
    let config = bitnet_oxidized::server::ServerConfig {
        max_batch_size: batch_size,
        max_queue_size: 64,
        timeout_seconds: 30,
    };
    let rt = tokio::runtime::Runtime::new().map_err(|e| anyhow::anyhow!("tokio runtime: {}", e))?;
    rt.block_on(bitnet_oxidized::server::run_server(
        model_path,
        None::<&std::path::Path>,
        port,
        config,
    ))
    .map_err(|e| anyhow::anyhow!("server: {}", e))
}

fn run_profile(model_path: &std::path::Path, iterations: usize) -> Result<()> {
    info!("Loading model from {:?}...", model_path);
    let model = if model_path.exists() {
        bitnet_oxidized::model::gguf::load_gguf(model_path)?
    } else {
        create_demo_model()
    };
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
    anyhow::bail!(
        "Quantize not yet implemented: input={:?} output={:?}. Use examples/quantize_model.rs with in-memory weights.",
        input,
        output
    )
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
