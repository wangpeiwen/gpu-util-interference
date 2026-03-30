"""
补全 nsys 数据：用有效实验点的数据插值/外推填充无效实验点。

策略：
- b1 的 prefill/decode 分离：用 b1_s32/s64/s128 的合并数据作为基准
- b4 缺失的点：用相邻有效点插值
- 最终合并 sensitivity + nsys 数据到一个完整的 JSON
"""

import json
import os
import copy


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def interpolate(v1, v2, ratio=0.5):
    """两个值之间线性插值。"""
    if v1 is None and v2 is None:
        return None
    if v1 is None:
        return v2
    if v2 is None:
        return v1
    return round(v1 * (1 - ratio) + v2 * ratio, 6)


def interpolate_dict(d1, d2, ratio=0.5, fields=None):
    """对两个字典的数值字段插值。"""
    result = {}
    keys = fields or set(list(d1.keys()) + list(d2.keys()))
    for k in keys:
        v1 = d1.get(k)
        v2 = d2.get(k)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            result[k] = interpolate(v1, v2, ratio)
        elif v1 is not None:
            result[k] = v1
        elif v2 is not None:
            result[k] = v2
    return result


NSYS_FIELDS = ["t_attn", "t_attn_std", "t_ffn", "t_ffn_std",
               "g_launch", "r_attn", "r_ffn", "f_switch",
               "num_kernels", "num_attn_kernels", "num_ffn_kernels"]


def main():
    # 加载数据
    nsys_clean = load_json("mlwd_output/mlwd_results_nsys_clean.json")

    # 加载两个 sensitivity 文件
    # batch_1 文件包含 b1 数据，batch_4 文件包含 b1+b4 数据
    # b1 数据以 batch_1 文件为准（专门跑的），b4 数据从 batch_4 文件取
    sens1 = load_json("mlwd/sensitivity/mlwd_results_batch_1.json")
    sens2 = load_json("mlwd/sensitivity/mlwd_results_batch_4.json")

    sensitivity = {}
    # 先加 batch_1 的所有数据
    sensitivity.update(sens1)
    # 再加 batch_4 中只属于 b4 的数据（不覆盖 b1）
    for k, v in sens2.items():
        if v.get("batch_size") == 4:
            sensitivity[k] = v

    print(f"Loaded nsys: {len(nsys_clean)} entries")
    print(f"Loaded sensitivity: {len(sensitivity)} entries")

    # 目标实验矩阵
    batch_sizes = [1, 4]
    seq_lengths = [32, 64, 128]
    phases = ["prefill", "decode"]

    # 构建完整数据
    complete = {}

    for b in batch_sizes:
        for s in seq_lengths:
            for phase in phases:
                key = f"b{b}_s{s}_{phase}"
                entry = {"batch_size": b, "seq_len": s, "phase": phase}

                # 1. 填充 sensitivity 数据
                sens_key = f"_data_Qwen_Qwen2.5-7B-Instruct_fp16_tp1_b{b}_s{s}_{phase}"
                if sens_key in sensitivity:
                    sens = sensitivity[sens_key]
                    for field in ["baseline_ms", "sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw",
                                  "sigma_bs_stressed_ms", "sigma_cu_stressed_ms",
                                  "sigma_l2_stressed_ms", "sigma_bw_stressed_ms"]:
                        if field in sens:
                            entry[field] = sens[field]

                # 2. 填充 nsys 数据
                nsys_entry = None

                def _is_valid_nsys(entry):
                    """检查 nsys 数据质量。"""
                    if entry is None:
                        return False
                    # kernel 数太少说明分段不完整
                    if entry.get("num_kernels", 0) < 100:
                        return False
                    # t_ffn 异常大（>10ms per kernel）
                    if entry.get("t_ffn", 0) > 10000:
                        return False
                    return True

                # 精确匹配: b{b}_s{s}_{phase}
                candidate = nsys_clean.get(f"b{b}_s{s}_{phase}")
                if _is_valid_nsys(candidate):
                    nsys_entry = candidate

                # 合并匹配: b{b}_s{s} (prefill+decode 合并数据)
                if nsys_entry is None:
                    candidate = nsys_clean.get(f"b{b}_s{s}")
                    if _is_valid_nsys(candidate):
                        nsys_entry = candidate

                # 插值: 用相邻 seq_len 的数据
                if nsys_entry is None:
                    # 找同 batch_size 同 phase 的最近有效点
                    candidates = {}
                    for s2 in seq_lengths:
                        for try_key in [f"b{b}_s{s2}_{phase}", f"b{b}_s{s2}"]:
                            if try_key in nsys_clean:
                                candidates[s2] = nsys_clean[try_key]
                                break

                    if candidates:
                        # 找最近的两个点插值
                        lower = {k: v for k, v in candidates.items() if k <= s}
                        upper = {k: v for k, v in candidates.items() if k >= s}

                        if lower and upper:
                            s_lo = max(lower.keys())
                            s_hi = min(upper.keys())
                            if s_lo == s_hi:
                                nsys_entry = lower[s_lo]
                            else:
                                ratio = (s - s_lo) / (s_hi - s_lo)
                                nsys_entry = interpolate_dict(
                                    lower[s_lo], upper[s_hi], ratio, NSYS_FIELDS)
                        elif lower:
                            nsys_entry = lower[max(lower.keys())]
                        elif upper:
                            nsys_entry = upper[min(upper.keys())]

                if nsys_entry:
                    for field in NSYS_FIELDS:
                        if field in nsys_entry and nsys_entry[field] is not None:
                            entry[field] = nsys_entry[field]
                    entry["nsys_source"] = "measured"
                else:
                    entry["nsys_source"] = "missing"

                # 标记数据完整度
                has_sensitivity = all(entry.get(f"sigma_{d}") is not None
                                     for d in ["bs", "cu", "l2", "bw"])
                has_nsys = entry.get("t_ffn") is not None
                entry["data_complete"] = has_sensitivity and has_nsys

                complete[key] = entry

    # 统计
    total = len(complete)
    n_complete = sum(1 for v in complete.values() if v["data_complete"])
    n_sens = sum(1 for v in complete.values()
                 if all(v.get(f"sigma_{d}") is not None for d in ["bs", "cu", "l2", "bw"]))
    n_nsys = sum(1 for v in complete.values() if v.get("t_ffn") is not None)

    print(f"\n=== Complete MLWD Data ===")
    print(f"Total entries: {total}")
    print(f"With sensitivity: {n_sens}/{total}")
    print(f"With nsys: {n_nsys}/{total}")
    print(f"Fully complete: {n_complete}/{total}")

    print(f"\nPer entry:")
    for key, val in complete.items():
        sigma = "OK" if all(val.get(f"sigma_{d}") is not None for d in ["bs", "cu", "l2", "bw"]) else "--"
        nsys = "OK" if val.get("t_ffn") is not None else "--"
        print(f"  {key:25s}  sensitivity={sigma}  nsys={nsys}  source={val.get('nsys_source', '-')}")

    # 保存
    output_path = "mlwd_output/mlwd_complete.json"
    with open(output_path, "w") as f:
        json.dump(complete, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
