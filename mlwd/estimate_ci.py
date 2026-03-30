"""
从模型结构理论计算 FLOPs，结合 profiler 时延估算 CI。

Qwen2.5-7B 结构：
- hidden_size: 3584
- num_layers: 28
- num_attention_heads: 28
- num_key_value_heads: 4 (GQA)
- intermediate_size: 18944
- head_dim: 128

Attention FLOPs per layer (prefill, seq_len=s, batch=b):
  QKV projection: 2 * b * s * h * (h + 2*kv_h*d) = 2 * b * s * 3584 * (3584 + 2*4*128)
  Attention score: 2 * b * num_heads * s * s * d = 2 * b * 28 * s^2 * 128
  Attention output: 2 * b * num_heads * s * d * d_model/num_heads... 简化为 QK^T + softmax + AV + output proj

FFN FLOPs per layer (Qwen2 用 SwiGLU):
  gate_proj: 2 * b * s * h * inter = 2 * b * s * 3584 * 18944
  up_proj:   2 * b * s * h * inter = 2 * b * s * 3584 * 18944
  down_proj: 2 * b * s * inter * h = 2 * b * s * 18944 * 3584
  Total FFN: 3 * 2 * b * s * 3584 * 18944

Usage:
    PYTHONPATH=. python mlwd/estimate_ci.py --profiler mlwd_output/mlwd_results_profiler.json --output mlwd_output/mlwd_results_ci.json
"""

import argparse
import json
import os


# Qwen2.5-7B 模型参数
HIDDEN_SIZE = 3584
NUM_LAYERS = 28
NUM_HEADS = 28
NUM_KV_HEADS = 4  # GQA
HEAD_DIM = 128
INTERMEDIATE_SIZE = 18944
VOCAB_SIZE = 152064


def compute_attn_flops(batch_size, seq_len, num_layers=NUM_LAYERS):
    """计算 Attention 总 FLOPs (prefill+decode 合并)。"""
    b, s = batch_size, seq_len
    h = HIDDEN_SIZE
    d = HEAD_DIM
    nh = NUM_HEADS
    nkv = NUM_KV_HEADS

    # QKV projection: Q(h->h) + K(h->kv_dim) + V(h->kv_dim)
    qkv_flops = 2 * b * s * h * (h + 2 * nkv * d)
    # QK^T: b * nh * s * s * d (每个 head)
    qk_flops = 2 * b * nh * s * s * d
    # Attention * V: b * nh * s * d * s
    av_flops = 2 * b * nh * s * s * d
    # Output projection: b * s * h * h
    out_flops = 2 * b * s * h * h

    per_layer = qkv_flops + qk_flops + av_flops + out_flops
    return per_layer * num_layers


def compute_ffn_flops(batch_size, seq_len, num_layers=NUM_LAYERS):
    """计算 FFN 总 FLOPs (SwiGLU: gate + up + down)。"""
    b, s = batch_size, seq_len
    h = HIDDEN_SIZE
    inter = INTERMEDIATE_SIZE

    # gate_proj: 2*b*s*h*inter, up_proj: 同, down_proj: 2*b*s*inter*h
    per_layer = 3 * 2 * b * s * h * inter
    return per_layer * num_layers


def main():
    parser = argparse.ArgumentParser(description="Estimate CI from model structure + profiler timing")
    parser.add_argument("--profiler", type=str, default="mlwd_output/mlwd_results_profiler.json")
    parser.add_argument("--output", type=str, default="mlwd_output/mlwd_results_ci.json")
    args = parser.parse_args()

    with open(args.profiler) as f:
        profiler_data = json.load(f)

    results = {}

    for key, pdata in profiler_data.items():
        b = pdata.get("batch_size")
        s = pdata.get("seq_len")
        if b is None or s is None:
            continue

        # 理论 FLOPs
        # 注意：profiler 采集了 num_runs 次推理，每次包含 prefill(s tokens) + decode(max_tokens tokens)
        # 简化：用 prefill 的 s 作为主要序列长度
        max_tokens = 32  # 默认 decode 长度
        # Prefill 阶段的 FLOPs
        attn_flops_prefill = compute_attn_flops(b, s)
        ffn_flops_prefill = compute_ffn_flops(b, s)
        # Decode 阶段的 FLOPs (每步 seq_len=1，共 max_tokens 步)
        attn_flops_decode = compute_attn_flops(b, 1) * max_tokens
        ffn_flops_decode = compute_ffn_flops(b, 1) * max_tokens

        total_attn_flops = attn_flops_prefill + attn_flops_decode
        total_ffn_flops = ffn_flops_prefill + ffn_flops_decode

        # 从 profiler 拿时延 (μs)
        attn_time_us = pdata.get("attn_time_us", 0)
        ffn_time_us = pdata.get("ffn_time_us", 0)
        num_runs = 3  # 默认

        # 每次 run 的 FLOPs
        attn_flops_per_run = total_attn_flops
        ffn_flops_per_run = total_ffn_flops

        # profiler 采集了 num_runs 次，时延是总和
        total_attn_flops_all = attn_flops_per_run * num_runs
        total_ffn_flops_all = ffn_flops_per_run * num_runs

        # V100 HBM2 带宽: 900 GB/s
        # CI = FLOPs / DRAM_bytes
        # DRAM_bytes 估算: 对于 memory-bound kernel, bytes ≈ time * bandwidth
        # 对于 compute-bound kernel, bytes < time * bandwidth
        # 更准确的方式: 直接用 FLOPs / time 得到 FLOP/s，再除以带宽得到 CI
        # CI = (FLOPs / time) / bandwidth = FLOPs / (time * bandwidth)
        V100_BW = 900e9  # bytes/s

        entry = {
            "batch_size": b,
            "seq_len": s,
            "attn_flops_theory": total_attn_flops,
            "ffn_flops_theory": total_ffn_flops,
            "attn_time_us": attn_time_us,
            "ffn_time_us": ffn_time_us,
        }

        if attn_time_us > 0:
            # CI = FLOPs / (time_s * bandwidth)
            attn_time_s = (attn_time_us / num_runs) * 1e-6
            entry["ci_attn"] = round(total_attn_flops / (attn_time_s * V100_BW), 4)
            entry["attn_tflops"] = round(total_attn_flops / attn_time_s / 1e12, 2)

        if ffn_time_us > 0:
            ffn_time_s = (ffn_time_us / num_runs) * 1e-6
            entry["ci_ffn"] = round(total_ffn_flops / (ffn_time_s * V100_BW), 4)
            entry["ffn_tflops"] = round(total_ffn_flops / ffn_time_s / 1e12, 2)

        print(f"\n{key}:")
        print(f"  Attn: {total_attn_flops:.2e} FLOPs, {attn_time_us/num_runs:.0f} μs/run"
              f" -> CI={entry.get('ci_attn', '-')}, {entry.get('attn_tflops', '-')} TFLOPS")
        print(f"  FFN:  {total_ffn_flops:.2e} FLOPs, {ffn_time_us/num_runs:.0f} μs/run"
              f" -> CI={entry.get('ci_ffn', '-')}, {entry.get('ffn_tflops', '-')} TFLOPS")

        results[key] = entry

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
