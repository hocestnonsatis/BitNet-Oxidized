# Models

## bitnet_b1_58-large (small, fast test) — recommended for quick runs

Model: [1bitLLM/bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large) — ~729M params, ~2.9 GB.

### Download and convert

```bash
# Download
.venv/bin/python scripts/download_hf_model.py --repo 1bitLLM/bitnet_b1_58-large --out models/bitnet_b1_58-large

# Convert to GGUF (~2 min)
.venv/bin/python scripts/convert_huggingface_to_gguf.py \
  --input models/bitnet_b1_58-large \
  --output models/bitnet_b1_58-large.gguf
```

### Run

```bash
cargo run --release -- info --model models/bitnet_b1_58-large.gguf
# With tokenizer for decoded text:
cargo run --release -- infer --model models/bitnet_b1_58-large.gguf \
  --prompt "Hello, my name is" --max-tokens 40 \
  --tokenizer models/bitnet_b1_58-large/tokenizer.json
# Without --tokenizer: output is token IDs
```

 For real text, use the HTTP server with the model’s tokenizer (see below).

---

## Llama3-8B-1.58-100B-tokens (Hugging Face)

Model: [HF1BitLLM/Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens)

### 1. Download

From the project root, with a Python venv that has `huggingface_hub`:

```bash
python -m venv .venv
.venv/bin/pip install huggingface_hub
.venv/bin/python scripts/download_hf_model.py
```

This writes to `models/Llama3-8B-1.58-100B-tokens/` (config, tokenizer, `model.safetensors` ~3.85 GB).

### 2. Convert to GGUF

The HF model uses GQA and sharded FFN; the converter expands tensors to the layout expected by bitnet-oxidized. Install deps and run (can take 10–15+ minutes and several GB RAM):

```bash
.venv/bin/pip install safetensors numpy
.venv/bin/python scripts/convert_huggingface_to_gguf.py \
  --input models/Llama3-8B-1.58-100B-tokens \
  --output models/Llama3-8B-1.58-100B-tokens.gguf
```

### 3. Run

```bash
# Info
cargo run --release -- info --model models/Llama3-8B-1.58-100B-tokens.gguf

# Inference (uses simple word tokenizer; for real text use the HTTP server with tokenizer.json)
cargo run --release -- infer --model models/Llama3-8B-1.58-100B-tokens.gguf --prompt "Hello" --max-tokens 20

# HTTP server (with tokenizer for /v1/completions and /v1/chat/completions)
cargo run --release -- serve --model models/Llama3-8B-1.58-100B-tokens.gguf --port 8080
# Then pass tokenizer path in code or use a wrapper that loads tokenizer.json from the same directory.
```

The `infer` command uses a naive tokenizer; for proper LLaMA tokenization use the server with `BitNetTokenizer::from_file("models/Llama3-8B-1.58-100B-tokens/tokenizer.json")` (see `src/server/mod.rs` for tokenizer path).
