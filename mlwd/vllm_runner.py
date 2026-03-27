"""
vLLM 离线推理封装。

精确控制 (batch_size, seq_len, phase) 进行推理，
支持作为独立脚本被 ncu/nsys 子进程调用。
"""

import argparse
import time
import json
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class InferenceResult:
    model: str
    phase: str
    batch_size: int
    seq_len: int
    latency_ms: float
    per_run_latencies_ms: List[float]


def create_synthetic_prompts(tokenizer, seq_len: int, batch_size: int) -> List[str]:
    """生成精确 token 长度的合成 prompt。"""
    # 用重复 token 构造精确长度的输入
    base_token = "hello "
    # 先编码一个长字符串，再截断到目标长度
    long_text = base_token * (seq_len * 2)
    token_ids = tokenizer.encode(long_text)[:seq_len]
    prompt = tokenizer.decode(token_ids)
    return [prompt] * batch_size


def run_prefill(llm, tokenizer, batch_size: int, seq_len: int,
                num_runs: int, warmup_runs: int) -> InferenceResult:
    """测量 Prefill 阶段时延：长 prompt + max_tokens=1。"""
    from vllm import SamplingParams

    prompts = create_synthetic_prompts(tokenizer, seq_len, batch_size)
    sampling_params = SamplingParams(max_tokens=1, temperature=0)

    # warmup
    for _ in range(warmup_runs):
        llm.generate(prompts, sampling_params)

    latencies = []
    for i in range(num_runs):
        start = time.perf_counter()
        llm.generate(prompts, sampling_params)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)
        print(f"  [Prefill] Run {i}: {elapsed_ms:.2f} ms (b={batch_size}, s={seq_len})")

    latencies.sort()
    mid = len(latencies) // 2
    median = latencies[mid] if num_runs % 2 == 1 else (latencies[mid - 1] + latencies[mid]) / 2

    return InferenceResult(
        model="", phase="prefill", batch_size=batch_size,
        seq_len=seq_len, latency_ms=median, per_run_latencies_ms=latencies,
    )


def run_decode(llm, tokenizer, batch_size: int, seq_len: int,
               num_runs: int, warmup_runs: int) -> InferenceResult:
    """测量 Decode 阶段时延：短 prompt + 生成 seq_len 个 token。"""
    from vllm import SamplingParams

    # 短 prompt 触发 decode
    short_prompt = "The"
    prompts = [short_prompt] * batch_size
    sampling_params = SamplingParams(max_tokens=seq_len, temperature=0)

    # warmup
    for _ in range(warmup_runs):
        llm.generate(prompts, SamplingParams(max_tokens=4, temperature=0))

    latencies = []
    for i in range(num_runs):
        start = time.perf_counter()
        llm.generate(prompts, sampling_params)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)
        print(f"  [Decode] Run {i}: {elapsed_ms:.2f} ms (b={batch_size}, gen={seq_len})")

    latencies.sort()
    mid = len(latencies) // 2
    median = latencies[mid] if num_runs % 2 == 1 else (latencies[mid - 1] + latencies[mid]) / 2

    return InferenceResult(
        model="", phase="decode", batch_size=batch_size,
        seq_len=seq_len, latency_ms=median, per_run_latencies_ms=latencies,
    )


def load_vllm_model(model_name: str, quantization: str = "fp16",
                     tp_degree: int = 1, gpu_memory_utilization: float = 0.9):
    """加载 vLLM 模型。"""
    import os
    os.environ["VLLM_USE_V1"] = "0"

    from vllm import LLM
    from transformers import AutoTokenizer

    dtype_map = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32",
                 "int8": "float16", "int4": "float16"}
    dtype = dtype_map.get(quantization, "float16")

    kwargs = {
        "model": model_name,
        "dtype": dtype,
        "tensor_parallel_size": tp_degree,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "enforce_eager": True,  # 禁用 CUDA graph，减少显存占用
    }
    if quantization == "int8":
        kwargs["quantization"] = "squeezellm"
    elif quantization == "int4":
        kwargs["quantization"] = "awq"

    llm = LLM(**kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return llm, tokenizer


def main():
    parser = argparse.ArgumentParser(description="vLLM MLWD Inference Runner")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quantization", type=str, default="fp16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--phase", type=str, required=True, choices=["prefill", "decode"])
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--warmup_runs", type=int, default=2)
    parser.add_argument("--gpu_mem", type=float, default=0.9,
                        help="vLLM gpu_memory_utilization (降低此值以适应显存不足)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="输出结果到 JSON 文件")
    args = parser.parse_args()

    print(f"Loading model: {args.model} ({args.quantization}, TP={args.tp})...")
    llm, tokenizer = load_vllm_model(args.model, args.quantization, args.tp, args.gpu_mem)
    print("Model loaded.")

    if args.phase == "prefill":
        result = run_prefill(llm, tokenizer, args.batch_size, args.seq_len,
                             args.num_runs, args.warmup_runs)
    else:
        result = run_decode(llm, tokenizer, args.batch_size, args.seq_len,
                            args.num_runs, args.warmup_runs)

    result.model = args.model
    print(f"\nMedian latency: {result.latency_ms:.2f} ms")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({
                "model": result.model,
                "phase": result.phase,
                "batch_size": result.batch_size,
                "seq_len": result.seq_len,
                "latency_ms": result.latency_ms,
                "per_run_latencies_ms": result.per_run_latencies_ms,
            }, f, indent=2)


if __name__ == "__main__":
    main()
