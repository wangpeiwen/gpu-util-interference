"""清理 nsys 结果，只保留有效条目。"""
import json

with open("mlwd_output/mlwd_results_nsys.json") as f:
    data = json.load(f)

clean = {}
for key, val in data.items():
    if key.startswith("segment_") or key == "all":
        continue
    if val.get("num_attn_kernels", 0) > 0 or val.get("num_ffn_kernels", 0) > 0:
        clean[key] = val

print(f"Kept {len(clean)} / {len(data)} entries:\n")
for k, v in clean.items():
    print(f"  {k}: {v.get('num_kernels')} kernels, "
          f"attn={v.get('num_attn_kernels')}, ffn={v.get('num_ffn_kernels')}, "
          f"t_attn={v.get('t_attn', '-')}, t_ffn={v.get('t_ffn', '-')}")

with open("mlwd_output/mlwd_results_nsys_clean.json", "w") as f:
    json.dump(clean, f, indent=2)

print(f"\nSaved to mlwd_output/mlwd_results_nsys_clean.json")
