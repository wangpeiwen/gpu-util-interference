import sqlite3

conn = sqlite3.connect('/tmp/mlwd_trace.sqlite')

str_ids = {}
for r in conn.execute("SELECT id, value FROM StringIds").fetchall():
    str_ids[r[0]] = r[1]

print("=== NVTX ranges (start, end) ===")
for r in conn.execute("SELECT text, start, end FROM NVTX_EVENTS ORDER BY start").fetchall():
    text = r[0]
    if isinstance(text, int):
        text = str_ids.get(text, str(text))
    print(f"  {text}: start={r[1]} end={r[2]} dur={r[2]-r[1] if r[2] else 'None'}")

print("\n=== Kernel time range ===")
row = conn.execute("SELECT MIN(start) as mn, MAX(end) as mx FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
print(f"  Kernel min_start={row[0]} max_end={row[1]}")

print("\n=== NVTX time range ===")
row2 = conn.execute("SELECT MIN(start) as mn, MAX(end) as mx FROM NVTX_EVENTS").fetchone()
print(f"  NVTX min_start={row2[0]} max_end={row2[1]}")

conn.close()
