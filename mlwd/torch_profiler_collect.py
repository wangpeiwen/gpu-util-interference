"""
使用 torch.profiler 采集 MLWD 第一层中 ncu 相关的特征。

不需要 ncu 权限，通过 PyTorch Profiler + CUPTI 采集：
- CI (Compute Intensity): 通过 FLOPs / DRAM bytes 估算
- IPC: 通过 profiler 的 kernel 统计估算
- kernel 级 FLOPs 和时延

Usage:
    PYTHONPATH=. python mlwd/torch_profiler_collect.py \
      --model /data/Qwen/Qwen2.5-7B-Instruct \
      --batch_sizes 1 4 --seq_lengths 32 64 128 \
      --output mlwd_output/mlwd_results_profiler.json
"""

import argparse
import json
import os
import time
import torch
from itertools import product

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlwd.profiling.kernel_classifier import classify_kernel, KernelCategory


def main():
    parser = argparse.ArgumentParser(description="Torch Profiler MLWD Collection")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quantization", type=str, default="fp16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[32])
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--warmup_runs", type=int, default=2)
    parser.add_argument("--output", type=str, default="mlwd_output/mlwd_results_profiler.json")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model}...")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(args.quantization, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch_dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print("Model loaded.\n")

    # Global warmup
    with torch.no_grad():
        dummy = tokenizer("Hello", return_tensors="pt").to("cuda")
        model.generate(**dummy, max_new_tokens=2, do_sample=False)
    torch.cuda.synchronize()

    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)

    for b, s in product(args.batch_sizes, args.seq_lengths):
        key = f"b{b}_s{s}"

        if key in results and results[key].get("ci_ffn") is not None:
            print(f"[{key}] SKIP (already collected)")
            continue

        print(f"\n{'='*50}")
        print(f"Profiling: {key}")
        print(f"{'='*50}")

        # 构造输入
        base_text = "hello " * (s * 2)
        token_ids = tokenizer.encode(base_text)[:s]
        prompt = tokenizer.decode(token_ids)
        # 构造 batch
        inputs = tokenizer([prompt] * b, return_tensors="pt", padding=True).to("cuda")

        # Warmup
        with torch.no_grad():
            for _ in range(args.warmup_runs):
                model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        torch.cuda.synchronize()

        # Profile
        all_events = []
        for run_i in range(args.num_runs):
            torch.cuda.synchronize()

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_flops=True,
                profile_memory=True,
                record_shapes=True,
            ) as prof:
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
                torch.cuda.synchronize()

            # 提取 kernel 级事件
            events = prof.key_averages()
            for evt in events:
                if evt.device_type == torch.autograd.DeviceType.CUDA and evt.count > 0:
                    all_events.append({
                        "name": evt.key,
                        "count": evt.count,
                        "cuda_time_us": evt.cuda_time_total,
                        "cuda_time_avg_us": evt.cuda_time_total / evt.count,
                        "flops": evt.flops if evt.flops else 0,
                        "cpu_time_us": evt.cpu_time_total,
                    })

            print(f"  run {run_i}: {len(events)} event types")

        # 分类聚合
        attn_flops = 0
        attn_time_us = 0
        attn_count = 0
        ffn_flops = 0
        ffn_time_us = 0
        ffn_count = 0
        total_flops = 0
        total_time_us = 0
        total_count = 0

        for evt in all_events:
            cat = classify_kernel(evt["name"])
            total_flops += evt["flops"]
            total_time_us += evt["cuda_time_us"]
            total_count += evt["count"]

            if cat == KernelCategory.ATTENTION:
                attn_flops += evt["flops"]
                attn_time_us += evt["cuda_time_us"]
                attn_count += evt["count"]
            elif cat == KernelCategory.FFN:
                ffn_flops += evt["flops"]
                ffn_time_us += evt["cuda_time_us"]
                ffn_count += evt["count"]

        # 估算 CI
        # V100 HBM2 带宽 900 GB/s，用 kernel 时间 * 带宽估算 DRAM bytes
        # CI = FLOPs / DRAM_bytes
        # 更准确的方式：用 profiler 报告的 FLOPs / (cuda_time * peak_bw * utilization)
        # 这里用简化估算：假设 memory-bound kernel 的带宽利用率 ~60%
        V100_BW_BYTES_PER_US = 900_000  # 900 GB/s = 900,000 MB/s = 900,000 bytes/μs...
        # 实际 900 GB/s = 900 * 1e9 bytes/s = 900,000 bytes/μs
        # 不对，900 GB/s = 900 * 1e9 / 1e6 bytes/μs = 900,000 bytes/μs

        entry = {
            "batch_size": b,
            "seq_len": s,
            "total_flops": total_flops,
            "total_cuda_time_us": total_time_us,
            "total_kernel_count": total_count,
            "attn_flops": attn_flops,
            "attn_cuda_time_us": attn_time_us,
            "attn_kernel_count": attn_count,
            "ffn_flops": ffn_flops,
            "ffn_cuda_time_us": ffn_time_us,
            "ffn_kernel_count": ffn_count,
        }

        # CI 估算：FLOPs / (time_us * peak_bandwidth * est_utilization)
        bw_util = 0.6  # 估算带宽利用率
        if attn_time_us > 0:
            est_attn_bytes = attn_time_us * V100_BW_BYTES_PER_US * bw_util
            entry["ci_attn"] = round(attn_flops / est_attn_bytes, 6) if est_attn_bytes > 0 else None
        if ffn_time_us > 0:
            est_ffn_bytes = ffn_time_us * V100_BW_BYTES_PER_US * bw_util
            entry["ci_ffn"] = round(ffn_flops / est_ffn_bytes, 6) if est_ffn_bytes > 0 else None

        # 直接可用的指标
        if attn_count > 0:
            entry["profiler_t_attn"] = round(attn_time_us / attn_count, 4)
        if ffn_count > 0:
            entry["profiler_t_ffn"] = round(ffn_time_us / ffn_count, 4)

        # 打印 top kernel
        print(f"\n  Attn: {attn_count} kernels, {attn_flops:.2e} FLOPs, {attn_time_us:.0f} μs")
        print(f"  FFN:  {ffn_count} kernels, {ffn_flops:.2e} FLOPs, {ffn_time_us:.0f} μs")
        print(f"  CI_attn={entry.get('ci_attn', '-')}, CI_ffn={entry.get('ci_ffn', '-')}")

        # Top 10 kernels by time
        sorted_events = sorted(all_events, key=lambda e: -e["cuda_time_us"])
        print(f"\n  Top 10 kernels by CUDA time:")
        for i, evt in enumerate(sorted_events[:10]):
            cat = classify_kernel(evt["name"])
            print(f"    {i+1}. [{cat.value:5s}] {evt['cuda_time_avg_us']:.1f}μs x{evt['count']} "
                  f"FLOPs={evt['flops']:.2e} | {evt['name'][:70]}")

        results[key] = entry

        # 立即保存
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
