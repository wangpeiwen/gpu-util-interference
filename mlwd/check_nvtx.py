import sqlite3

conn = sqlite3.connect('/tmp/mlwd_trace.sqlite')

rows = conn.execute("""
    SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start
""").fetchall()

gaps = []
for i in range(1, len(rows)):
    gap = rows[i][0] - rows[i-1][1]
    gaps.append((i, gap))

# 按 gap 大小排序，取 top 30
top_gaps = sorted(gaps, key=lambda x: -x[1])[:30]
print("Top 30 gaps:")
for idx, gap in top_gaps:
    print(f"  between kernel {idx-1}-{idx}: {gap/1e6:.1f} ms")

# 如果用 top 11 gap 分段（12 实验点 = 11 个分界）
# 加上模型加载前的 gap = 12 个分界
print(f"\nTotal kernels: {len(rows)}")

# 取 top N gap 作为分界，看分出的段各有多少 kernel
n_segments = 13  # warmup + 12 实验点
split_indices = sorted([idx for idx, gap in top_gaps[:n_segments-1]])
print(f"\nSplit at indices: {split_indices}")

segments = []
prev = 0
for idx in split_indices:
    segments.append((prev, idx, idx - prev))
    prev = idx
segments.append((prev, len(rows), len(rows) - prev))

print(f"\nSegments ({len(segments)}):")
for i, (start, end, count) in enumerate(segments):
    print(f"  seg {i}: kernels [{start}, {end}) count={count}")

conn.close()
