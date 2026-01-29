# Microsoft BitNet reference code

Code copied from **https://github.com/microsoft/BitNet** for reference only.

- **Not compiled or executed** in this project.
- Use for algorithm reference: ternary quantization, TL1/TL2 preprocessing, LUT kernels, HF→GGUF conversion logic.
- Original license: see [BitNet LICENSE](https://github.com/microsoft/BitNet/blob/main/LICENSE).

## Contents

| Path | Description |
|------|-------------|
| `include/ggml-bitnet.h` | GGML BitNet API (bitnet_tensor_extra, mul_mat, TL1/TL2) |
| `src/ggml-bitnet-lut.cpp` | GGML integration: LUT matmul, TL1/TL2 init/compute |
| `utils/preprocess-huggingface-bitnet.py` | Quantize HF weights to ternary (quant_weight_fp16) |
| `utils/convert-helper-bitnet.py` | Orchestration: preprocess → convert → quantize |
| `utils/bitnet_tl1_tl2_reference.py` | Extracted TL1/TL2 preprocessing logic (process_tl1, preprocess_weights_tl2, etc.) |
| `preset_kernels/bitnet_b1_58-large/` | Kernel config (kernel_config_tl1.ini, kernel_config_tl2.ini) |

## Key algorithms (for BitNet-Oxidized)

1. **Weight quantization** (BitnetModel / preprocess-huggingface-bitnet.py):
   ```python
   s = 1 / weight.abs().mean().clamp(min=1e-5)
   result = (weight * s).round().clamp(-1, 1) / s
   ```
2. **Ternary packing**: TL1 uses hi*3+lo, +4, then uint8 pack; process_tl1 reshapes for tiled LUT.
3. **GGUF types**: TL1 (2-bit), TL2 (2-bit), F32, F16; scales stored per-tensor for dequant.
