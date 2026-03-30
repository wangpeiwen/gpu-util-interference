"""
使用 vLLM 内置 profiler 采集 kernel 级 FLOPs 数据。

vLLM 的 start_profile/stop_profile 在 EngineCore 子进程内部
启动 torch.profiler，能捕获到实际的 CUDA kernel。

Usage:
    PYTHONPATH=. python mlwd/vllm_profiler_collect.py \
      --model /data/Qwen/Qwen2.5-7B-Instruct \
      --batch_sizes 1 4 --seq_lengths 32 64 128
"""

import argparse
import json
import os
import glob
import time
import torch
from itertools import product

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlwd.profiling.kernel_classifier import classify_kernel, KernelCategory


def parse_chrome_trace(trace_path):
    """解析 vLLM profiler 输出的 Chrome trace JSON。"""
    with open(trace_path) as f:
        trace = json.load(f)

    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])

    kernels = []
    for evt in events:
        if evt.get("cat") == "kernel" or evt.get("cat") == "cuda_runtime":
            kernels.append({
                "name": evt.get("name", ""),
                "dur_us": evt.get("dur", 0),
                "args": evt.get("args", {}),
            })
        # PyTorch profiler 格式
        elif evt.get("cat") == "cuda_runtime" or "cuda" in str(evt.get("cat", "")):
            kernels.append({
                "name": evt.get("name", ""),
                "dur_us": evt.get("dur", 0),
                "args": evt.get("args", {}),
            })

    return kernels


def parse_profiler_output(profile_dir):
    """解析 vLLM profiler 输出目录中的 trace 文件。"""
    # 查找最新的 trace 文件
    patterns = [
        os.path.join(profile_dir, "*.json"),
        os.path.join(profile_dir, "*.pt.trace.json"),
        os.path.join(profile_dir, "**/*.json"),
    ]

    trace_files = []
    for pat in patterns:
        trace_files.extend(glob.glob(pat, recursive=True))

    if not trace_files:
        print(f"  No trace files found in {profile_dir}")
        return []

    # 用最新的文件
    trace_files.sort(key=os.path.getmtime, reverse=True)
    trace_path = trace_files[0]
    print(f"  Parsing trace: {trace_path}")

    return parse_chrome_trace(trace_path)


def main():
    parser = argparse.ArgumentParser(description="vLLM Profiler MLWD Collection")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quantization", type=str, default="fp16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[32])
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--warmup_runs", type=int, default=2)
    parser.add_argument("--profile_dir", type=str, default="/tmp/vllm_profile")
    parser.add_argument("--output", type=str, default="mlwd_output/mlwd_results_profiler.json")
    args = parser.parse_args()

    os.environ["VLLM_USE_V1"] = "0"
    os.makedirs(args.profile_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading model: {args.model}...")
    llm = LLM(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        enforce_eager=True,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": args.profile_dir,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Model loaded.\n")

    # Global warmup
    llm.generate(["Hello"], SamplingParams(max_tokens=1, temperature=0))

    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)

    for b, s in product(args.batch_sizes, args.seq_lengths):
        key = f"b{b}_s{s}"

        if key in results and results[key].get("profiler_done"):
            print(f"[{key}] SKIP")
            continue

        print(f"\n{'='*50}")
        print(f"Profiling: {key}")
        print(f"{'='*50}")

        # 构造输入
        base_text = "hello " * (s * 2)
        token_ids = tokenizer.encode(base_text)[:s]
        prompt = tokenizer.decode(token_ids)
        prompts = [prompt] * b
        sp = SamplingParams(max_tokens=args.max_tokens, temperature=0)

        # Warmup
        for _ in range(args.warmup_runs):
            llm.generate(prompts, sp)

        # 清理旧 trace
        for f in glob.glob(os.path.join(args.profile_dir, "*")):
            os.remove(f)

        # 开启 profiler
        llm.start_profile()

        # 推理
        for run_i in range(args.num_runs):
            llm.generate(prompts, sp)
            print(f"  run {run_i} done")

        # 停止 profiler
        llm.stop_profile()

        # 等待 trace 文件写入
        time.sleep(2)

        # 解析 trace
        trace_files = glob.glob(os.path.join(args.profile_dir, "**/*.json"), recursive=True)
        trace_files += glob.glob(os.path.join(args.profile_dir, "**/*.json.gz"), recursive=True)
        trace_files += glob.glob(os.path.join(args.profile_dir, "**/*.pt.trace.json"), recursive=True)
        trace_files += glob.glob(os.path.join(args.profile_dir, "**/*.pt.trace.json.gz"), recursive=True)

        print(f"  Trace files: {trace_files}")

        if not trace_files:
            print(f"  No trace files found, listing dir:")
            for f in glob.glob(os.path.join(args.profile_dir, "**/*"), recursive=True):
                print(f"    {f} ({os.path.getsize(f)} bytes)")
            results[key] = {"profiler_done": True, "error": "no trace files"}
        else:
            # 解析最新的 trace
            trace_files.sort(key=os.path.getmtime, reverse=True)
            trace_path = trace_files[0]
            print(f"  Parsing: {trace_path} ({os.path.getsize(trace_path)} bytes)")

            try:
                import gzip
                if trace_path.endswith(".gz"):
                    with gzip.open(trace_path, "rt") as f:
                        trace = json.load(f)
                else:
                    with open(trace_path) as f:
                        trace = json.load(f)

                events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
                print(f"  Total events: {len(events)}")

                # 统计 CUDA kernel
                cuda_events = [e for e in events if "cuda" in str(e.get("cat", "")).lower()
                               or e.get("cat") == "kernel"]
                print(f"  CUDA events: {len(cuda_events)}")

                # 统计所有 event categories
                cats = {}
                for e in events:
                    cat = e.get("cat", "unknown")
                    cats[cat] = cats.get(cat, 0) + 1
                print(f"  Event categories: {dict(sorted(cats.items(), key=lambda x: -x[1])[:10])}")

                # 提取 kernel 级数据
                attn_time = 0
                attn_flops = 0
                attn_count = 0
                ffn_time = 0
                ffn_flops = 0
                ffn_count = 0
                total_time = 0
                total_flops = 0

                for evt in events:
                    name = evt.get("name", "")
                    dur = evt.get("dur", 0)
                    flops = evt.get("args", {}).get("flops", 0)
                    cat = classify_kernel(name)

                    if dur > 0:
                        total_time += dur
                        total_flops += flops

                        if cat == KernelCategory.ATTENTION:
                            attn_time += dur
                            attn_flops += flops
                            attn_count += 1
                        elif cat == KernelCategory.FFN:
                            ffn_time += dur
                            ffn_flops += flops
                            ffn_count += 1

                V100_BW_BYTES_PER_US = 900_000  # 900 GB/s

                entry = {
                    "batch_size": b,
                    "seq_len": s,
                    "profiler_done": True,
                    "total_events": len(events),
                    "cuda_events": len(cuda_events),
                    "attn_count": attn_count,
                    "attn_flops": attn_flops,
                    "attn_time_us": attn_time,
                    "ffn_count": ffn_count,
                    "ffn_flops": ffn_flops,
                    "ffn_time_us": ffn_time,
                }

                # CI 估算
                bw_util = 0.6
                if attn_flops > 0 and attn_time > 0:
                    est_bytes = attn_time * V100_BW_BYTES_PER_US * bw_util
                    entry["ci_attn"] = round(attn_flops / est_bytes, 6)
                if ffn_flops > 0 and ffn_time > 0:
                    est_bytes = ffn_time * V100_BW_BYTES_PER_US * bw_util
                    entry["ci_ffn"] = round(ffn_flops / est_bytes, 6)

                print(f"\n  Attn: {attn_count} kernels, {attn_flops:.2e} FLOPs, {attn_time:.0f} μs")
                print(f"  FFN:  {ffn_count} kernels, {ffn_flops:.2e} FLOPs, {ffn_time:.0f} μs")
                print(f"  CI_attn={entry.get('ci_attn', '-')}, CI_ffn={entry.get('ci_ffn', '-')}")

                results[key] = entry

            except Exception as e:
                print(f"  Parse error: {e}")
                results[key] = {"profiler_done": True, "error": str(e)}

        # 保存
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
