"""
LLM Inference Interference Benchmark

Measures how GPU resource contention affects LLM inference latency.
Uses the project's fma_fp32_ilp4 kernel as a controllable interference source.

Usage:
    # Basic usage with a small model
    python3 run_llm_interference.py --model gpt2

    # With custom interference intensity
    python3 run_llm_interference.py --model gpt2 --num_tb 132 --iters_interf 500000

    # With MPS for SM-level isolation (run as root or with MPS permissions)
    python3 run_llm_interference.py --model gpt2 --use_mps --mps_pct 50
"""

import argparse
import time
import threading
import os
import subprocess
from ctypes import CDLL

import torch


def load_model(model_name, dtype):
    """Load a HuggingFace causal LM model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="cuda"
    )
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, prompt, max_new_tokens, num_runs):
    """Run LLM inference and return per-run latencies (ms)."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()

    latencies = []
    tokens_generated = []

    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        num_tokens = outputs.shape[1] - input_len
        latencies.append(elapsed_ms)
        tokens_generated.append(num_tokens)

        print(f"  Run {i}: {elapsed_ms:.2f} ms, {num_tokens} tokens, "
              f"{num_tokens / (elapsed_ms / 1000.0):.1f} tokens/s")

    latencies.sort()
    mid = len(latencies) // 2
    median_lat = latencies[mid] if num_runs % 2 == 1 else (latencies[mid - 1] + latencies[mid]) / 2
    avg_tokens = sum(tokens_generated) / len(tokens_generated)

    print(f"  Median latency: {median_lat:.2f} ms, "
          f"Median throughput: {avg_tokens / (median_lat / 1000.0):.1f} tokens/s")
    return median_lat


def run_interference_kernel(c_funcs, num_tb, num_threads, iters_interf, runs_interf):
    """Launch the fma_fp32_ilp4 interference kernel via CTypes."""
    c_funcs.run_fp32_fma_kernel(num_tb, num_threads, iters_interf, runs_interf)


def start_mps(pct):
    """Start MPS daemon with given active thread percentage."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"
    subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)
    print(f"MPS started with {pct}% active threads")


def stop_mps():
    """Stop MPS daemon."""
    subprocess.run("echo quit | nvidia-cuda-mps-control", shell=True, check=False)
    print("MPS stopped")


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Interference Benchmark")
    # Model config
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--prompt", type=str,
                        default="The future of artificial intelligence is",
                        help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=5, help="Number of inference runs")

    # Interference kernel config
    parser.add_argument("--num_tb", type=int, default=132,
                        help="Thread blocks for interference kernel (set to num SMs)")
    parser.add_argument("--num_threads", type=int, default=128,
                        help="Threads per block for interference kernel")
    parser.add_argument("--iters_interf", type=int, default=300000,
                        help="Iterations for interference kernel (controls duration)")
    parser.add_argument("--runs_interf", type=int, default=4,
                        help="Number of interference kernel launches")

    # MPS config
    parser.add_argument("--use_mps", action="store_true", help="Use MPS for SM partitioning")
    parser.add_argument("--mps_pct", type=int, default=50,
                        help="MPS active thread percentage per process")

    # Shared library path
    parser.add_argument("--shared_lib", type=str,
                        default="./../build/mm_pytorch/libpython_interface.so",
                        help="Path to the compiled shared library")

    args = parser.parse_args()

    # Load interference kernel library
    c_funcs = CDLL(args.shared_lib)
    print(f"Loaded interference library: {args.shared_lib}")

    # Load LLM
    print(f"Loading model: {args.model} ({args.dtype})...")
    model, tokenizer = load_model(args.model, args.dtype)
    print("Model loaded.\n")

    launch_cfg = f"({args.num_tb} blocks, {args.num_threads} threads, {args.iters_interf} iters)"

    if args.use_mps:
        start_mps(args.mps_pct)

    try:
        # Phase 1: Baseline - LLM inference alone
        print("=" * 60)
        print("Phase 1: LLM inference ALONE")
        print("=" * 60)
        lat_alone = run_inference(model, tokenizer, args.prompt, args.max_new_tokens, args.runs)

        # Phase 2: Interference kernel alone
        print(f"\n{'=' * 60}")
        print(f"Phase 2: Interference kernel ALONE {launch_cfg}")
        print("=" * 60)
        run_interference_kernel(c_funcs, args.num_tb, args.num_threads,
                                args.iters_interf, args.runs_interf)

        # Phase 3: Colocated - LLM + interference kernel
        print(f"\n{'=' * 60}")
        print(f"Phase 3: LLM + Interference kernel COLOCATED {launch_cfg}")
        print("=" * 60)

        interf_thread = threading.Thread(
            target=run_interference_kernel,
            args=(c_funcs, args.num_tb, args.num_threads,
                  args.iters_interf, args.runs_interf),
        )
        interf_thread.start()
        lat_coloc = run_inference(model, tokenizer, args.prompt, args.max_new_tokens, args.runs)
        interf_thread.join()

        # Summary
        slowdown = ((lat_coloc - lat_alone) / lat_alone) * 100
        print(f"\n{'=' * 60}")
        print("RESULTS SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Alone median latency:    {lat_alone:.2f} ms")
        print(f"  Colocated median latency:{lat_coloc:.2f} ms")
        print(f"  Slowdown:                {slowdown:+.1f}%")
        print(f"  Interference config:     {launch_cfg}")

    finally:
        if args.use_mps:
            stop_mps()


if __name__ == "__main__":
    main()
