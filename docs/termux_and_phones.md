# Running on Termux / phones

## Do not run HF → GGUF conversion on your phone

Converting a Hugging Face BitNet model (safetensors) to GGUF **loads the full model into memory** and does heavy processing. On a phone this can:

- Use 1.5 GB+ RAM and cause **all apps to crash** or the system to kill processes
- Max out the CPU and make the device unresponsive

**Do not run** `convert_huggingface_to_gguf.py` or `convert_hf_to_gguf_stdlib.py` on Termux or other memory-constrained devices.

## Safe workflow on a phone

1. **Convert on a desktop or server** (PC, cloud, GitHub Actions, etc.):
   - Download the HF model there (or use an existing clone).
   - Run the conversion:
     ```bash
     python3 scripts/convert_huggingface_to_gguf.py \
       --input /path/to/bitnet-b1.58-2B-4T \
       --output bitnet-b1.58-2B-4T.gguf
     ```
   - Copy only the resulting **`.gguf` file** to your phone (e.g. via USB, cloud storage, or `scp`).

2. **On the phone (Termux)** use the copied GGUF for inference:
   ```bash
   cargo run -- infer --model /path/to/bitnet-b1.58-2B-4T.gguf --prompt "Hello"
   # Or with tokenizer for decoded text:
   cargo run -- infer --model /path/to/bitnet-b1.58-2B-4T.gguf --prompt "Hello" --tokenizer models/bitnet-b1.58-2B-4T/tokenizer.json
   ```

3. **Downloading the HF model on the phone** (e.g. with `curl` for `config.json` + `model.safetensors`) is fine for **storage** only. Do not run the conversion step on the device.

## Optional: pre-built GGUF from Microsoft

Microsoft provides a pre-converted GGUF at [microsoft/bitnet-b1.58-2B-4T-gguf](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) (`ggml-model-i2_s.gguf`). That file is for **bitnet.cpp** and may use different tensor names than bitnet-oxidized. If we add support for that format, you could download only the GGUF on the phone (no conversion). For now, prefer “convert on desktop, copy .gguf” as above.
