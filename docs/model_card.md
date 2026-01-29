# BitNet-Oxidized Model Card

## Model Details

- **Architecture**: BitNet b1.58 (ternary-weight transformer)
- **Parameters**: Configurable (demo: ~125K; production models: 125M–3B+)
- **Precision**: 1.58-bit (ternary weights: -1, 0, +1), 2-bit packed
- **Training Data**: Depends on the specific checkpoint; not defined by this framework
- **Training Procedure**: See quantization and conversion scripts; QAT support in framework

## Intended Use

- **Primary intended uses**: Text generation, chat, code completion, research
- **Primary intended users**: Researchers, developers, production inference services
- **Out-of-scope uses**: Cryptographic or safety-critical decisions without additional safeguards

## Performance

| Model size (approx) | Memory (approx) | Speed (tokens/s, CPU) |
|--------------------|-----------------|------------------------|
| 125M               | ~32 MB          | ~1000                  |
| 1B                 | ~256 MB         | ~500                   |
| 3B                 | ~768 MB         | ~200                   |

*Actual numbers depend on hardware (CPU cores, SIMD), sequence length, and batch size.*

## Bias, Risks, and Limitations

- Output quality and safety depend on the underlying checkpoint and tokenizer.
- Ternary quantization can reduce accuracy compared to full-precision models.
- This card describes the **framework** (BitNet-Oxidized); deployers should add model-specific bias and risk documentation for each checkpoint.

## How to Use

### Rust library

```rust
use bitnet_oxidized::model::gguf;
use bitnet_oxidized::{BitNetTokenizer, TextGenerator};

let model = gguf::load_gguf("model.gguf")?;
let tokenizer = BitNetTokenizer::from_file("tokenizer.json")?;
let generator = TextGenerator::new(model);

let prompt_ids = tokenizer.encode("Hello, world!")?;
let output_ids = generator.generate_greedy(&prompt_ids, prompt_ids.len() + 100)?;
let text = tokenizer.decode(&output_ids)?;
```

### HTTP API (serve command)

```bash
bitnet-oxidized serve --model model.gguf --port 8080
# POST /v1/completions or /v1/chat/completions
```

See [Deployment Guide](deployment_guide.md) for Docker and Kubernetes.
