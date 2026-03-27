"""
Kernel 名称分类器：按正则匹配将 CUDA kernel 分类为 Attention / FFN / Other。

用于 Nsight Compute 和 Nsight Systems 采集结果的后处理。
"""

import re
from enum import Enum
from typing import Optional


class KernelCategory(Enum):
    ATTENTION = "attention"
    FFN = "ffn"
    OTHER = "other"


# vLLM 在 V100 上常见的 kernel 名称模式
ATTENTION_PATTERNS = [
    re.compile(r"flash_attn", re.IGNORECASE),
    re.compile(r"fmha", re.IGNORECASE),
    re.compile(r"paged_attention", re.IGNORECASE),
    re.compile(r"attention.*kernel", re.IGNORECASE),
    re.compile(r"cutlass.*attention", re.IGNORECASE),
    re.compile(r"multi_head_attention", re.IGNORECASE),
    re.compile(r"scaled_dot_product", re.IGNORECASE),
    re.compile(r"flash.*fwd", re.IGNORECASE),
    re.compile(r"flash.*bwd", re.IGNORECASE),
]

FFN_PATTERNS = [
    re.compile(r"gemm", re.IGNORECASE),
    re.compile(r"cublas.*gemm", re.IGNORECASE),
    re.compile(r"cutlass.*gemm", re.IGNORECASE),
    re.compile(r"sgemm", re.IGNORECASE),
    re.compile(r"hgemm", re.IGNORECASE),
    re.compile(r"volta.*gemm", re.IGNORECASE),
    re.compile(r"sm70.*gemm", re.IGNORECASE),
    re.compile(r"linear", re.IGNORECASE),
    re.compile(r"mlp.*kernel", re.IGNORECASE),
    re.compile(r"fc_kernel", re.IGNORECASE),
]

# 排除模式：这些 kernel 虽然匹配 FFN 但实际属于 Attention 内部
ATTENTION_GEMM_PATTERNS = [
    re.compile(r"attn.*gemm", re.IGNORECASE),
    re.compile(r"attention.*gemm", re.IGNORECASE),
    re.compile(r"qkv.*gemm", re.IGNORECASE),
]


def classify_kernel(kernel_name: str) -> KernelCategory:
    """将 kernel 名称分类为 Attention / FFN / Other。"""
    # 先检查是否是 Attention 内部的 GEMM
    for pat in ATTENTION_GEMM_PATTERNS:
        if pat.search(kernel_name):
            return KernelCategory.ATTENTION

    # 检查 Attention 模式
    for pat in ATTENTION_PATTERNS:
        if pat.search(kernel_name):
            return KernelCategory.ATTENTION

    # 检查 FFN 模式
    for pat in FFN_PATTERNS:
        if pat.search(kernel_name):
            return KernelCategory.FFN

    return KernelCategory.OTHER


def classify_kernels(kernel_names: list) -> dict:
    """批量分类，返回 {name: category} 映射。"""
    return {name: classify_kernel(name) for name in kernel_names}
