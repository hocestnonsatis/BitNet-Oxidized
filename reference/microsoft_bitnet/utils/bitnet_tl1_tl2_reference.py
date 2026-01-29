# Extracted from https://github.com/microsoft/BitNet (utils/convert-hf-to-gguf-bitnet.py)
# TL1/TL2 preprocessing and BitnetModel weight quantization - reference only.
# Not runnable as-is: depends on configparser (kernel_config.ini), numpy; used inside full converter.

from typing import Tuple
import numpy as np


def weight_quant(weight):
    """BitnetModel.weight_quant: scale by 1/mean(abs), round to {-1,0,1}, then / scale."""
    # From BitnetModel.modify_tensors: result = (weight * s).round().clamp(-1, 1) / s
    weight = weight.astype(np.float32)
    s = 1.0 / np.clip(np.abs(weight).mean(), 1e-5, None)
    result = np.clip(np.round(weight * s), -1, 1) / s
    return result


def process_tl1(weight, BM, BY, bm, by, M, K):
    weight = weight.reshape((M, K // 2)).astype(np.uint8)
    weight = weight.reshape((M // BM, BM, K // 2)).transpose(0, 2, 1)
    weight = weight.reshape((M // BM, K // BY, BY // 2, BM)).transpose(0, 1, 3, 2)
    weight = weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 2)).transpose(0, 1, 2, 4, 3)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, by // 2, bm)).transpose(0, 1, 2, 3, 5, 4)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, bm // 16, 16, by // 2)).transpose(0, 1, 2, 3, 4, 6, 5)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, bm // 16, by // 4, 4 // 2, 16)).transpose(0, 1, 2, 3, 4, 5, 7, 6)
    weight = weight.reshape((M * K // 16 // 4, 16, 4 // 2))
    weight_0 = weight[:, :, 0] << 4
    weight_1 = weight[:, :, 1]
    weight = weight_0 + weight_1
    return weight


def preprocess_weights_tl1(w: np.ndarray, bits=2, g=4) -> np.ndarray:
    """Ternary (-1,0,1) -> hi*3+lo, +4, pack to uint8; then process_tl1 reshape for tiled LUT."""
    from configparser import ConfigParser
    config = ConfigParser()
    M, K = w.shape
    weight = np.where(np.abs(w) < 1e-6, 0, w).astype(np.float32)
    weight = np.sign(weight)
    weight_num = np.prod(weight.shape)
    config.read('preset_kernels/bitnet_b1_58-large/kernel_config_tl1.ini')
    BM = BY = bm = -1
    for kernel in config.sections():
        if int(config.get(kernel, 'm')) == M and int(config.get(kernel, 'k')) == K:
            BM = int(config.get(kernel, 'bm'))
            BY = int(config.get(kernel, 'bk'))
            bm = int(config.get(kernel, 'bmm'))
            by = 256 // bm
            break
    if BM == -1:
        raise NotImplementedError
    weight = np.reshape(weight, (weight_num // 2, 2))
    hi_weight = np.multiply(np.split(weight, 2, axis=1)[0], 3)
    lo_weight = np.split(weight, 2, axis=1)[1]
    weight = np.reshape((hi_weight + lo_weight), weight_num // 2)
    weight = weight + 4
    weight = np.reshape(weight, (M, K // 2)).astype(np.uint8)
    weight = process_tl1(weight, BM, BY, bm, by, M, K)
    return weight


def preprocess_two_weights_tl2(M, K, weight_num, BM, BY, bm, by, weight, final_weight):
    weight = np.reshape(weight, (weight_num // 2, 2))
    hi_weight = np.multiply(np.split(weight, 2, axis=1)[0], 3)
    lo_weight = np.split(weight, 2, axis=1)[1]
    weight = np.reshape((hi_weight + lo_weight), weight_num // 2)
    weight = weight + 4
    weight = np.reshape(weight, (M, K // 2)).astype(np.uint8)
    weight = weight.reshape((M // BM, BM, K // 2)).transpose(0, 2, 1)
    weight = weight.reshape((M // BM, K // BY, BY // 2, BM)).transpose(0, 1, 3, 2)
    weight = weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 2)).transpose(0, 1, 2, 4, 3)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, by // 2, bm)).transpose(0, 1, 2, 3, 5, 4)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, bm, by // 2))
    weight_0 = weight[:, :, :, :, :, 0] << 4
    weight_1 = weight[:, :, :, :, :, 1]
    weight = weight_0 + weight_1
    weight = weight.reshape((M * K // bm // by, bm // 8, 8))
    weight[:, [0, 1, 2, 3], :] = weight[:, [0, 2, 1, 3], :]
    weight = weight.reshape(M * K // bm // by, bm)
    for i in range(weight.shape[0]):
        final_weight.append(weight[i, :])


def preprocess_three_weights_tl2(M, K, weight_num, BM, BY, bm, by, weight, final_weight):
    weight = np.reshape(weight, (weight_num // 3, 3))
    split_weights = np.split(weight, 3, axis=1)
    first_weight = np.multiply(split_weights[0], 9)
    second_weight = np.multiply(split_weights[1], 3)
    third_weight = split_weights[2]
    weight = np.reshape((first_weight + second_weight + third_weight), weight_num // 3)
    sign_weight = np.sign(weight) + 2
    sign_weight = np.where(sign_weight > 1, 0, sign_weight)
    weight = np.abs(weight)
    weight = np.reshape(weight, (M, K // 3)).astype(np.uint8)
    sign_weight = np.reshape(sign_weight, (M, K // 3)).astype(np.uint8)
    weight = weight.reshape((M // BM, BM, K // 3)).transpose(0, 2, 1)
    weight = weight.reshape((M // BM, K // BY, BY // 3, BM)).transpose(0, 1, 3, 2)
    weight = weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 3)).transpose(0, 1, 2, 4, 3)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, by // 3, bm)).transpose(0, 1, 2, 3, 5, 4)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, bm, by // 3))
    weight_0 = weight[:, :, :, :, :, 0] << 4
    weight_1 = weight[:, :, :, :, :, 1]
    weight = weight_0 + weight_1
    weight = weight.reshape((M * K // bm // by, bm // 8, 8))
    weight[:, [0, 1, 2, 3], :] = weight[:, [0, 2, 1, 3], :]
    weight = weight.reshape(M * K // bm // by, bm)
    for i in range(weight.shape[0]):
        final_weight.append(weight[i, :])
    sign_weight = sign_weight.reshape((M // BM, BM, K // 3)).transpose(0, 2, 1)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BY // 3, BM)).transpose(0, 1, 3, 2)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 3)).transpose(0, 1, 2, 4, 3)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, BY // (by * 4), by // 3 * 4, bm)).transpose(0, 1, 2, 3, 5, 4)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, BY // (by * 4), bm, by // 3 * 4)).transpose(0, 1, 2, 3, 5, 4)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, BY // (by * 4), by // 3 * 8, bm // 2)).astype(np.uint16)
    combine_weight = np.zeros((M // BM, K // BY, BM // bm, BY // (by * 4), bm // 2), dtype=np.uint16)
    for i in range(16):
        temp_weight = sign_weight[:, :, :, :, i, :] << 15 - i
        combine_weight += temp_weight
    combine_weight = combine_weight.view(np.uint8)
    combine_weight = combine_weight.reshape((M * K // bm // (by * 4)), bm)
    for i in range(combine_weight.shape[0]):
        final_weight.append(combine_weight[i, :])


def preprocess_weights_tl2(w: np.ndarray, bits=2, g=4) -> np.ndarray:
    from configparser import ConfigParser
    config = ConfigParser()
    M, K = w.shape
    weight = np.where(np.abs(w) < 1e-6, 0, w).astype(np.float32)
    weight = np.sign(weight)
    weight_num = np.prod(weight.shape)
    config.read('preset_kernels/bitnet_b1_58-large/kernel_config_tl2.ini')
    BM = BY = bm = -1
    for kernel in config.sections():
        if int(config.get(kernel, 'm')) == M and int(config.get(kernel, 'k')) == K:
            BM = int(config.get(kernel, 'bm'))
            BY = int(config.get(kernel, 'bk'))
            bm = int(config.get(kernel, 'bmm'))
            by = 192 // bm
            break
    if BM == -1:
        raise NotImplementedError
    if weight.shape[1] % BY != 0:
        slice_k_idx = weight.shape[1] - weight.shape[1] % BY
        slice_weights = np.split(weight, [slice_k_idx], axis=1)
        three_weight = slice_weights[0]
        two_weight = slice_weights[1]
    else:
        three_weight = weight
    final_weight = []
    preprocess_three_weights_tl2(three_weight.shape[0], three_weight.shape[1],
                                 three_weight.shape[0] * three_weight.shape[1],
                                 BM, BY, bm, by, three_weight, final_weight)
    if weight.shape[1] % BY != 0:
        preprocess_two_weights_tl2(two_weight.shape[0], two_weight.shape[1],
                                   two_weight.shape[0] * two_weight.shape[1],
                                   BM, 32, 32, 4, two_weight, final_weight)
    weight = np.array(final_weight, dtype=np.uint8).reshape(-1)
    weight = np.pad(weight, (0, (K - 256) * M // 3 * 5 // 8 + 256 * M // 2 * 4 // 8 - weight.shape[0]),
                    mode='constant', constant_values=0)
    return weight


def transform_to_tl1(x: np.ndarray) -> Tuple[np.ndarray, float]:
    scale = np.max(np.abs(x))
    res = preprocess_weights_tl1(x)
    return res, scale


def transform_to_tl2(x: np.ndarray) -> Tuple[np.ndarray, float]:
    scale = np.max(np.abs(x))
    res = preprocess_weights_tl2(x)
    return res, scale
