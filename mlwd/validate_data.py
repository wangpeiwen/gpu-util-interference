"""
验证 mlwd_complete.json 数据合理性。

检查项：
1. 数据完整性：所有字段非空
2. 值域合理性：σ >= 0, 0 <= r <= 1, t > 0 等
3. 趋势一致性：decode baseline 应随 seq_len 增长，prefill σ 应高于 decode 等
4. 异常值检测
"""

import json
import sys


def load_data(path="mlwd_output/mlwd_complete.json"):
    with open(path) as f:
        return json.load(f)


def check_completeness(data):
    """检查所有字段非空。"""
    required = [
        "baseline_ms", "sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw",
        "t_ffn", "g_launch", "r_ffn", "f_switch",
    ]
    issues = []
    for key, val in data.items():
        missing = [f for f in required if val.get(f) is None]
        if missing:
            issues.append(f"  {key}: missing {missing}")
    return issues


def check_ranges(data):
    """检查值域合理性。"""
    issues = []
    for key, val in data.items():
        # σ 应 >= 0
        for dim in ["bs", "cu", "l2", "bw"]:
            s = val.get(f"sigma_{dim}")
            if s is not None and s < 0:
                issues.append(f"  {key}: sigma_{dim}={s} < 0")

        # r_attn + r_ffn 应 <= 1
        r_a = val.get("r_attn", 0)
        r_f = val.get("r_ffn", 0)
        if r_a + r_f > 1.01:
            issues.append(f"  {key}: r_attn({r_a}) + r_ffn({r_f}) = {r_a+r_f} > 1")

        # baseline > 0
        bl = val.get("baseline_ms")
        if bl is not None and bl <= 0:
            issues.append(f"  {key}: baseline_ms={bl} <= 0")

        # t_ffn > 0
        tf = val.get("t_ffn")
        if tf is not None and tf <= 0:
            issues.append(f"  {key}: t_ffn={tf} <= 0")

        # stressed > baseline
        for dim in ["bs", "cu", "l2", "bw"]:
            stressed = val.get(f"sigma_{dim}_stressed_ms")
            if stressed is not None and bl is not None and stressed < bl:
                issues.append(f"  {key}: sigma_{dim}_stressed_ms({stressed}) < baseline({bl})")

    return issues


def check_trends(data):
    """检查趋势一致性。"""
    issues = []

    for b in [1, 4]:
        # decode baseline 应随 seq_len 单调增长
        decode_baselines = []
        for s in [32, 64, 128]:
            key = f"b{b}_s{s}_decode"
            if key in data and data[key].get("baseline_ms"):
                decode_baselines.append((s, data[key]["baseline_ms"]))

        for i in range(1, len(decode_baselines)):
            if decode_baselines[i][1] < decode_baselines[i-1][1]:
                issues.append(
                    f"  b{b} decode baseline not monotonic: "
                    f"s={decode_baselines[i-1][0]}({decode_baselines[i-1][1]:.1f}ms) > "
                    f"s={decode_baselines[i][0]}({decode_baselines[i][1]:.1f}ms)")

        # prefill σ 通常高于 decode σ（prefill 计算密集，更敏感）
        for s in [32, 64, 128]:
            p_key = f"b{b}_s{s}_prefill"
            d_key = f"b{b}_s{s}_decode"
            if p_key in data and d_key in data:
                for dim in ["cu", "l2", "bw"]:
                    p_sigma = data[p_key].get(f"sigma_{dim}")
                    d_sigma = data[d_key].get(f"sigma_{dim}")
                    if p_sigma is not None and d_sigma is not None:
                        if p_sigma < d_sigma:
                            issues.append(
                                f"  b{b}_s{s}: prefill sigma_{dim}({p_sigma:.3f}) "
                                f"< decode sigma_{dim}({d_sigma:.3f}) [unusual]")

    return issues


def check_nsys_anomalies(data):
    """检查 nsys 数据异常。"""
    issues = []

    for key, val in data.items():
        # b1 的 prefill 和 decode nsys 数据完全相同（因为用了合并数据）
        # 这是已知限制，标记出来
        b = val.get("batch_size")
        phase = val.get("phase")
        s = val.get("seq_len")

        if b == 1 and phase == "prefill":
            d_key = f"b{b}_s{s}_decode"
            if d_key in data:
                d_val = data[d_key]
                if (val.get("t_ffn") == d_val.get("t_ffn") and
                    val.get("g_launch") == d_val.get("g_launch")):
                    issues.append(
                        f"  b{b}_s{s}: prefill/decode nsys 数据相同 "
                        f"(合并 trace 未区分阶段)")

        # num_attn_kernels = 0 但 r_attn > 0 或反之
        n_attn = val.get("num_attn_kernels", 0)
        r_attn = val.get("r_attn", 0)
        if n_attn == 0 and r_attn > 0:
            issues.append(f"  {key}: num_attn_kernels=0 but r_attn={r_attn}")
        if n_attn > 0 and r_attn == 0:
            issues.append(f"  {key}: num_attn_kernels={n_attn} but r_attn=0")

        # t_ffn 异常大（>10000 μs = 10ms 单个 kernel 很少见）
        tf = val.get("t_ffn")
        if tf is not None and tf > 10000:
            issues.append(f"  {key}: t_ffn={tf:.1f}μs 异常大 (>10ms per kernel)")

    return issues


def main():
    data = load_data()
    print(f"Loaded {len(data)} entries\n")

    all_ok = True

    print("=== 1. 数据完整性 ===")
    issues = check_completeness(data)
    if issues:
        print("\n".join(issues))
        all_ok = False
    else:
        print("  OK: 所有必要字段完整\n")

    print("=== 2. 值域合理性 ===")
    issues = check_ranges(data)
    if issues:
        print("\n".join(issues))
        all_ok = False
    else:
        print("  OK: 所有值在合理范围内\n")

    print("=== 3. 趋势一致性 ===")
    issues = check_trends(data)
    if issues:
        print("\n".join(issues))
    else:
        print("  OK: 趋势一致\n")

    print("=== 4. nsys 数据异常 ===")
    issues = check_nsys_anomalies(data)
    if issues:
        print("\n".join(issues))
    else:
        print("  OK: 无异常\n")

    if all_ok:
        print("=== 验证通过 ===")
    else:
        print("=== 存在问题，请检查 ===")


if __name__ == "__main__":
    main()
