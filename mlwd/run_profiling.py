"""
独立 Profiling 脚本：模型只加载一次，循环跑所有实验点。

用 nsys 或 ncu 包装本脚本运行，避免每个实验点重新加载模型。
每个实验点用 NVTX marker 标记，方便从 trace 中区分。

注意：vLLM V1 引擎在子进程中执行 GPU 操作，nsys 需要加
--trace-fork-before-exec=true 才能捕获 CUDA kernel 数据。

Usage:
    # 直接运行（验证推理正常）
    python mlwd/run_profiling.py --model /data/Qwen/Qwen2.5-7B-Instruct --batch_sizes 1 --seq_lengths 32

    # nsys 采集（必须加 --trace-fork-before-exec=true 跟踪子进程）
    /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
      -o /tmp/mlwd_trace --trace cuda,nvtx \
      --trace-fork-before-exec=true --cuda-graph-trace=node \
      --sample none --cpuctxsw none --force-overwrite true \
      python mlwd/run_profiling.py --model /data/Qwen/Qwen2.5-7B-Instruct \
        --batch_sizes 1 4 --seq_lengths 32 64 128

    # nsys 导出 SQLite + 查看 kernel 统计
    /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys export \
      --type sqlite --output /tmp/mlwd_trace.sqlite --force-overwrite true /tmp/mlwd_trace.nsys-rep
    /opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys stats \
      --report cuda_gpu_kern_sum --force-export=true /tmp/mlwd_trace.nsys-rep

    # ncu 采集（建议单个 phase，减少时间）
    ncu --csv --metrics <metrics> --replay-mode application \
      --launch-skip 500 --launch-count 50 --log-file /tmp/ncu_out.csv \
      python mlwd/run_profiling.py --model /data/Qwen/Qwen2.5-7B-Instruct \
        --batch_sizes 1 --seq_lengths 32 --phases prefill
"""

import argparse
import json
import os
import time
from itertools import product

import torch


def main():
    parser = argparse.ArgumentParser(description="MLWD Profiling Runner")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quantization", type=str, default="fp16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[32])
    parser.add_argument("--phases", type=str, nargs="+", default=["prefill", "decode"],
                        choices=["prefill", "decode"])
    parser.add_argument("--num_runs", type=int, default=3,
                        help="每个实验点的推理次数")
    parser.add_argument("--warmup_runs", type=int, default=2)
    parser.add_argument("--output_meta", type=str, default=None,
                        help="输出元数据 JSON（记录每个实验点的 kernel 范围）")
    args = parser.parse_args()

    os.environ["VLLM_USE_V1"] = "0"

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # 加载模型（只加载一次）
    print(f"Loading model: {args.model}...")
    dtype_map = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
    llm = LLM(
        model=args.model,
        dtype=dtype_map.get(args.quantization, "float16"),
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Model loaded.\n")

    # 全局 warmup
    print("Global warmup...")
    warmup_sp = SamplingParams(max_tokens=1, temperature=0)
    llm.generate(["Hello"], warmup_sp)
    torch.cuda.synchronize()
    print("Warmup done.\n")

    meta = {}  # 记录每个实验点的元数据

    for b, s, phase in product(args.batch_sizes, args.seq_lengths, args.phases):
        key = f"b{b}_s{s}_{phase}"
        print(f"{'='*50}")
        print(f"Profiling: {key}")
        print(f"{'='*50}")

        # 构造输入
        if phase == "prefill":
            base_text = "hello " * (s * 2)
            token_ids = tokenizer.encode(base_text)[:s]
            prompt = tokenizer.decode(token_ids)
            prompts = [prompt] * b
            sp = SamplingParams(max_tokens=1, temperature=0)
        else:
            prompts = ["The"] * b
            sp = SamplingParams(max_tokens=s, temperature=0)

        # per-point warmup
        for _ in range(args.warmup_runs):
            llm.generate(prompts, sp)
        torch.cuda.synchronize()

        # 在 GPU 端制造 gap：sync + sleep + 空 kernel，确保 nsys trace 中有明确分界
        time.sleep(1.0)
        torch.cuda.synchronize()

        # NVTX marker + 推理
        latencies = []
        for i in range(args.num_runs):
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push(f"{key}_run{i}")

            t0 = time.perf_counter()
            llm.generate(prompts, sp)
            torch.cuda.synchronize()
            lat = (time.perf_counter() - t0) * 1000.0

            torch.cuda.nvtx.range_pop()

            latencies.append(lat)
            print(f"  run {i}: {lat:.2f} ms")

        latencies.sort()
        mid = len(latencies) // 2
        median = latencies[mid] if len(latencies) % 2 == 1 else (latencies[mid-1] + latencies[mid]) / 2
        print(f"  median: {median:.2f} ms\n")

        # 实验点结束后在 GPU 端制造 gap
        torch.cuda.synchronize()
        time.sleep(1.0)
        torch.cuda.synchronize()

        meta[key] = {
            "batch_size": b,
            "seq_len": s,
            "phase": phase,
            "median_ms": round(median, 4),
            "all_ms": [round(l, 4) for l in latencies],
        }

    # 保存元数据
    if args.output_meta:
        with open(args.output_meta, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata saved to {args.output_meta}")

    print("\nProfiling complete.")


if __name__ == "__main__":
    main()
