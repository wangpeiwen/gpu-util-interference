"""
MLWD 离线采集主编排脚本。

采集结果存储为 JSON 文件，每个实验点采集完一个维度就立即写入。

Usage:
    # 完整采集（仅 sensitivity，ncu/nsys 暂不可用）
    python -m mlwd.collect_all --model /data/Qwen/Qwen2.5-7B-Instruct --stage sensitivity

    # 指定 batch_size 和 seq_len
    python -m mlwd.collect_all --model /data/Qwen/Qwen2.5-7B-Instruct --stage sensitivity --batch_sizes 1 --seq_lengths 32

    # 完整矩阵
    python -m mlwd.collect_all --model /data/Qwen/Qwen2.5-7B-Instruct --stage sensitivity --batch_sizes 1 4 16 --seq_lengths 32 64 128 512 2048
"""

import argparse
import os
import json
import time
import threading
from pathlib import Path
from itertools import product

from .config import ExperimentConfig, StressKernelConfig
from .sensitivity.stress_kernels import StressKernels


def get_vllm_runner_path() -> str:
    return str(Path(__file__).parent / "vllm_runner.py")


def _make_key(model, quant, tp, b, s, phase):
    """生成实验点的唯一 key。"""
    model_short = model.replace("/", "_")
    return f"{model_short}_{quant}_tp{tp}_b{b}_s{s}_{phase}"


def _load_results(output_dir: str) -> dict:
    """加载已有的采集结果。"""
    results_file = os.path.join(output_dir, "mlwd_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    return {}


def _save_results(output_dir: str, results: dict):
    """保存采集结果到 JSON。"""
    results_file = os.path.join(output_dir, "mlwd_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _update_point(output_dir: str, key: str, updates: dict):
    """更新单个实验点的部分字段并立即写入。"""
    results = _load_results(output_dir)
    if key not in results:
        results[key] = {}
    results[key].update(updates)
    _save_results(output_dir, results)


def run_sensitivity_stage(config: ExperimentConfig, model: str, quant: str,
                          tp: int, shared_lib_path: str, output_dir: str):
    """Stage 3: 四维干扰敏感度采集，每个维度独立写入。"""
    stress_kernels = StressKernels(shared_lib_path)

    from .vllm_runner import load_vllm_model, create_synthetic_prompts

    print(f"  [Sensitivity] Loading model: {model}...")
    llm, tokenizer = load_vllm_model(model, quant, tp)
    print(f"  [Sensitivity] Model loaded.")

    from vllm import SamplingParams

    results = _load_results(output_dir)
    stress_config = config.stress

    dims = [
        ("sigma_bs", lambda: stress_kernels.run_bs_stress(
            stress_config.bs_num_tb, stress_config.bs_num_threads,
            stress_config.bs_num_itrs, 1)),
        ("sigma_cu", lambda: stress_kernels.run_cu_stress(
            stress_config.cu_num_tb, stress_config.cu_num_threads,
            stress_config.cu_num_itrs, 1)),
        ("sigma_l2", lambda: stress_kernels.run_l2_stress(
            stress_config.l2_num_tb, stress_config.l2_num_threads,
            stress_config.l2_num_itrs, stress_config.l2_num_bytes, 1)),
        ("sigma_bw", lambda: stress_kernels.run_bw_stress(
            stress_config.bw_num_tb, stress_config.bw_num_threads,
            stress_config.bw_num_itrs, stress_config.bw_num_bytes, 1)),
    ]

    for b in config.batch_sizes:
        for s in config.seq_lengths:
            for phase in config.phases:
                key = _make_key(model, quant, tp, b, s, phase)
                existing = results.get(key, {})

                print(f"\n  [Sensitivity] === b={b}, s={s}, {phase} ===")

                # 构造推理函数（用默认参数捕获当前值，避免闭包陷阱）
                if phase == "prefill":
                    prompts = create_synthetic_prompts(tokenizer, s, b)
                    sp = SamplingParams(max_tokens=1, temperature=0)
                    def run_fn(_p=prompts, _sp=sp):
                        return llm.generate(_p, _sp)
                else:
                    short_prompts = ["The"] * b
                    sp = SamplingParams(max_tokens=s, temperature=0)
                    def run_fn(_p=short_prompts, _sp=sp):
                        return llm.generate(_p, _sp)

                # 测量基线
                if "baseline_ms" not in existing:
                    print(f"  [Sensitivity] Measuring baseline...")
                    # warmup
                    for _ in range(config.warmup_runs):
                        run_fn()

                    latencies = []
                    for i in range(config.sensitivity_num_runs):
                        t0 = time.perf_counter()
                        run_fn()
                        lat = (time.perf_counter() - t0) * 1000.0
                        latencies.append(lat)
                        print(f"    baseline run {i}: {lat:.2f} ms")

                    latencies.sort()
                    mid = len(latencies) // 2
                    baseline = latencies[mid] if len(latencies) % 2 == 1 else (latencies[mid-1] + latencies[mid]) / 2
                    print(f"  [Sensitivity] Baseline median: {baseline:.2f} ms")

                    _update_point(output_dir, key, {
                        "model": model, "quantization": quant, "tp": tp,
                        "batch_size": b, "seq_len": s, "phase": phase,
                        "baseline_ms": round(baseline, 4),
                        "baseline_all_ms": [round(l, 4) for l in latencies],
                    })
                else:
                    baseline = existing["baseline_ms"]
                    print(f"  [Sensitivity] Baseline (cached): {baseline:.2f} ms")

                # 逐维度测量
                for dim_name, stress_fn in dims:
                    if dim_name in existing and existing[dim_name] is not None:
                        print(f"  [Sensitivity] Skip {dim_name} (already collected: {existing[dim_name]:.4f})")
                        continue

                    print(f"  [Sensitivity] Measuring {dim_name}...")

                    try:
                        stop_event = threading.Event()
                        def _bg_stress(_fn=stress_fn, _stop=stop_event):
                            while not _stop.is_set():
                                _fn()

                        stress_thread = threading.Thread(target=_bg_stress, daemon=True)
                        stress_thread.start()

                        # warmup under stress
                        for _ in range(config.warmup_runs):
                            run_fn()

                        # measure
                        latencies = []
                        for i in range(config.sensitivity_num_runs):
                            t0 = time.perf_counter()
                            run_fn()
                            lat = (time.perf_counter() - t0) * 1000.0
                            latencies.append(lat)
                            print(f"    {dim_name} run {i}: {lat:.2f} ms")

                        stop_event.set()
                        stress_thread.join(timeout=30)

                        latencies.sort()
                        mid = len(latencies) // 2
                        stressed = latencies[mid] if len(latencies) % 2 == 1 else (latencies[mid-1] + latencies[mid]) / 2

                        sigma = max((stressed - baseline) / baseline, 0.0)
                        print(f"  [Sensitivity] {dim_name} = {sigma:.4f} (baseline={baseline:.2f}ms, stressed={stressed:.2f}ms)")

                        _update_point(output_dir, key, {
                            dim_name: round(sigma, 6),
                            f"{dim_name}_stressed_ms": round(stressed, 4),
                            f"{dim_name}_all_ms": [round(l, 4) for l in latencies],
                        })
                        # 刷新 existing
                        existing = _load_results(output_dir).get(key, {})

                    except Exception as e:
                        print(f"  [Sensitivity] {dim_name} FAILED: {e}")
                        _update_point(output_dir, key, {dim_name: None, f"{dim_name}_error": str(e)})
                        existing = _load_results(output_dir).get(key, {})


def main():
    parser = argparse.ArgumentParser(description="MLWD Offline Collection Orchestrator")
    parser.add_argument("--model", type=str, nargs="+",
                        help="模型名称 (可指定多个)")
    parser.add_argument("--quantization", type=str, nargs="+", default=["fp16"])
    parser.add_argument("--tp", type=int, nargs="+", default=[1])
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=None)
    parser.add_argument("--stage", type=str, nargs="+",
                        choices=["ncu", "nsys", "sensitivity", "all"],
                        default=["all"])
    parser.add_argument("--output_dir", type=str, default="mlwd_output",
                        help="输出目录")
    parser.add_argument("--shared_lib", type=str,
                        default=str(Path(__file__).parent.parent / "build" / "mm_pytorch" / "libpython_interface.so"))
    args = parser.parse_args()

    config = ExperimentConfig()
    if args.model:
        config.models = args.model
    config.quantizations = args.quantization
    config.tp_degrees = args.tp
    if args.batch_sizes:
        config.batch_sizes = args.batch_sizes
    if args.seq_lengths:
        config.seq_lengths = args.seq_lengths

    stages = set(args.stage)
    run_all = "all" in stages

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"MLWD Offline Collection")
    print(f"  Models: {config.models}")
    print(f"  Batch sizes: {config.batch_sizes}")
    print(f"  Seq lengths: {config.seq_lengths}")
    print(f"  Stages: {stages}")
    print(f"  Total points: {config.total_points()}")
    print(f"  Output: {args.output_dir}/")
    print()

    for model, framework, quant, tp in config.iter_deployment_configs():
        print(f"{'='*60}")
        print(f"Config: {model} | {quant} | TP={tp}")
        print(f"{'='*60}")

        if run_all or "sensitivity" in stages:
            print("\n--- Interference Sensitivity ---")
            run_sensitivity_stage(config, model, quant, tp,
                                  args.shared_lib, args.output_dir)

        if run_all or "ncu" in stages:
            print("\n--- Nsight Compute (ncu) ---")
            print("  [NCU] TODO: ncu stage not yet integrated with JSON output")

        if run_all or "nsys" in stages:
            print("\n--- Nsight Systems (nsys) ---")
            print("  [NSYS] TODO: nsys stage not yet integrated with JSON output")

    # 打印汇总
    results = _load_results(args.output_dir)
    print(f"\nCollection complete. {len(results)} experiment points saved to {args.output_dir}/mlwd_results.json")


if __name__ == "__main__":
    main()
