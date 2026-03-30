import sqlite3
import statistics

conn = sqlite3.connect('/tmp/mlwd_trace.sqlite')

# kernel gaps 分布
rows = conn.execute("""
    SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start
""").fetchall()

gaps = []
for i in range(1, len(rows)):
    gap = rows[i][0] - rows[i-1][1]
    gaps.append(gap)

gaps_sorted = sorted(gaps)
print(f"Total kernels: {len(rows)}")
print(f"Total gaps: {len(gaps)}")
print(f"Gap percentiles (ns):")
for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
    idx = int(len(gaps_sorted) * p / 100)
    print(f"  p{p}: {gaps_sorted[idx]:,}")
print(f"  max: {gaps_sorted[-1]:,}")

# 大 gap 的位置
print(f"\nGaps > 10ms:")
for i, g in enumerate(gaps):
    if g > 10_000_000:
        print(f"  between kernel {i} and {i+1}: {g/1e6:.1f} ms")

print(f"\nGaps > 1ms (count): {sum(1 for g in gaps if g > 1_000_000)}")
print(f"Gaps > 10ms (count): {sum(1 for g in gaps if g > 10_000_000)}")
print(f"Gaps > 100ms (count): {sum(1 for g in gaps if g > 100_000_000)}")
print(f"Gaps > 1s (count): {sum(1 for g in gaps if g > 1_000_000_000)}")

conn.close()
