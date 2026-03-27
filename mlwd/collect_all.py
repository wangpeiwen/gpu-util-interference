"""
MLWD 离线采集主编排脚本。

串联 ncu profiling → nsys profiling → 干扰敏感度采集三个阶段，
支持分阶段运行和断点续采。

Usage:
    # 完整采集
    python -m mlwd.collect_all --model meta-llama/Llama-2-7b-hf

    # 仅采集 ncu
    python -m mlwd.collect_all --model meta-llama/Llama-2-7b-hf --stage ncu

    # 仅采集敏感度
    python -m mlwd.collect_all --model meta-llama/Llama-2-7b-hf --stage sensitivity
"""

import argparse
import os
import sys
import json
from pathlib import Path

from .config import ExperimentConfig, StressKernelConfig
from .storage.mlwd_schema import DeploymentConfig, OperatorProfile
from .storage.mlwd_db import MLWDDatabase
from .profiling.ncu_collector import run_ncu_profile
from .profiling.nsys_collector import run_nsys_profile
from .sensitivity.stress_kernels import StressKernels
from .sensitivity.sensitivity_collector import collect_sensitivity

def get_vllm_runner_path() -> str:
    return str(Path(__file__).parent / "vllm_runner.py")


def run_ncu_stage(db: MLWDDatabase, config: ExperimentConfig,
                  config_id: int, model: str, quant: str, tp: int):
    """Stage 1: Nsight Compute 采集 CI, L2, IPC。"""
    vllm_runner = get_vllm_runner_path()

    for b in config.batch_sizes:
        for s in config.seq_lengths:
            for phase in config.phases:
                if db.is_collected(config_id, b, s, phase, "ncu"):
                    print(f"  [NCU] Skip (already collected): b={b}, s={s}, {phase}")
                    continue

                print(f"\n  [NCU] Collecting: b={b}, s={s}, {phase}")
                ncu_result = run_ncu_profile(
                    model=model, quantization=quant, tp_degree=tp,
                    batch_size=b, seq_len=s, phase=phase,
                    vllm_runner_path=vllm_runner,
                    launch_count=config.ncu_num_runs,
                )

                profile = OperatorProfile(
                    config_id=config_id, batch_size=b, seq_len=s, phase=phase,
                    ci_attn=ncu_result.ci_attn, ci_ffn=ncu_result.ci_ffn,
                    l2_attn=ncu_result.l2_attn, l2_ffn=ncu_result.l2_ffn,
                    ipc=ncu_result.ipc,
                )
                db.upsert_profile(profile)


def run_nsys_stage(db: MLWDDatabase, config: ExperimentConfig,
                   config_id: int, model: str, quant: str, tp: int):
    """Stage 2: Nsight Systems 采集执行模式特征。"""
    vllm_runner = get_vllm_runner_path()

    for b in config.batch_sizes:
        for s in config.seq_lengths:
            for phase in config.phases:
                if db.is_collected(config_id, b, s, phase, "nsys"):
                    print(f"  [NSYS] Skip (already collected): b={b}, s={s}, {phase}")
                    continue

                print(f"\n  [NSYS] Collecting: b={b}, s={s}, {phase}")
                nsys_result = run_nsys_profile(
                    model=model, quantization=quant, tp_degree=tp,
                    batch_size=b, seq_len=s, phase=phase,
                    vllm_runner_path=vllm_runner,
                )

                profile = OperatorProfile(
                    config_id=config_id, batch_size=b, seq_len=s, phase=phase,
                    t_attn=nsys_result.t_attn, t_ffn=nsys_result.t_ffn,
                    g_launch=nsys_result.g_launch,
                    r_attn=nsys_result.r_attn, r_ffn=nsys_result.r_ffn,
                    f_switch=nsys_result.f_switch,
                    t_attn_std=nsys_result.t_attn_std, t_ffn_std=nsys_result.t_ffn_std,
                )
                db.upsert_profile(profile)


def run_sensitivity_stage(db: MLWDDatabase, config: ExperimentConfig,
                          config_id: int, model: str, quant: str, tp: int,
                          shared_lib_path: str):
    """Stage 3: 四维干扰敏感度采集。"""
    stress_kernels = StressKernels(shared_lib_path)

    # 需要在进程内加载 vLLM 模型
    from .vllm_runner import load_vllm_model, create_synthetic_prompts, run_prefill, run_decode

    print(f"  [Sensitivity] Loading model: {model}...")
    llm, tokenizer = load_vllm_model(model, quant, tp)
    print(f"  [Sensitivity] Model loaded.")

    from vllm import SamplingParams

    for b in config.batch_sizes:
        for s in config.seq_lengths:
            for phase in config.phases:
                if db.is_collected(config_id, b, s, phase, "sensitivity"):
                    print(f"  [Sensitivity] Skip: b={b}, s={s}, {phase}")
                    continue

                print(f"\n  [Sensitivity] Collecting: b={b}, s={s}, {phase}")

                # 构造推理函数
                if phase == "prefill":
                    prompts = create_synthetic_prompts(tokenizer, s, b)
                    sp = SamplingParams(max_tokens=1, temperature=0)
                    run_fn = lambda: llm.generate(prompts, sp)
                else:
                    short_prompts = ["The"] * b
                    sp = SamplingParams(max_tokens=s, temperature=0)
                    run_fn = lambda: llm.generate(short_prompts, sp)

                sens_result = collect_sensitivity(
                    run_inference_fn=run_fn,
                    stress_kernels=stress_kernels,
                    stress_config=config.stress,
                    num_runs=config.sensitivity_num_runs,
                    warmup_runs=config.warmup_runs,
                )

                profile = OperatorProfile(
                    config_id=config_id, batch_size=b, seq_len=s, phase=phase,
                    sigma_bs=sens_result.sigma_bs, sigma_cu=sens_result.sigma_cu,
                    sigma_l2=sens_result.sigma_l2, sigma_bw=sens_result.sigma_bw,
                    baseline_latency_ms=sens_result.baseline_latency_ms,
                )
                db.upsert_profile(profile)


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
                        default=["all"],
                        help="要运行的采集阶段")
    parser.add_argument("--db_path", type=str, default="mlwd_data.db")
    parser.add_argument("--shared_lib", type=str,
                        default=str(Path(__file__).parent.parent / "build" / "mm_pytorch" / "libpython_interface.so"))
    args = parser.parse_args()

    # 构建实验配置
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

    print(f"MLWD Offline Collection")
    print(f"  Models: {config.models}")
    print(f"  Quantizations: {config.quantizations}")
    print(f"  TP degrees: {config.tp_degrees}")
    print(f"  Batch sizes: {config.batch_sizes}")
    print(f"  Seq lengths: {config.seq_lengths}")
    print(f"  Stages: {stages}")
    print(f"  Total experiment points: {config.total_points()}")
    print(f"  DB: {args.db_path}")
    print()

    db = MLWDDatabase(args.db_path)

    try:
        for model, framework, quant, tp in config.iter_deployment_configs():
            dep_cfg = DeploymentConfig(
                model=model, framework=framework,
                quantization=quant, tp_degree=tp,
            )
            config_id = db.upsert_config(dep_cfg)
            print(f"{'='*60}")
            print(f"Config: {model} | {quant} | TP={tp} (config_id={config_id})")
            print(f"{'='*60}")

            if run_all or "ncu" in stages:
                print("\n--- Stage 1: Nsight Compute ---")
                run_ncu_stage(db, config, config_id, model, quant, tp)

            if run_all or "nsys" in stages:
                print("\n--- Stage 2: Nsight Systems ---")
                run_nsys_stage(db, config, config_id, model, quant, tp)

            if run_all or "sensitivity" in stages:
                print("\n--- Stage 3: Interference Sensitivity ---")
                run_sensitivity_stage(db, config, config_id, model, quant, tp,
                                      args.shared_lib)

        # 导出汇总
        all_data = db.export_all()
        print(f"\nCollection complete. {len(all_data)} records in database.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
