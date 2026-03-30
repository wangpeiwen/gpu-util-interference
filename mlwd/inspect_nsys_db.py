"""
查看 nsys 导出的 SQLite 数据库结构和内容。

Usage:
    python mlwd/inspect_nsys_db.py /tmp/mlwd_trace.sqlite
"""

import sqlite3
import sys


def inspect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 所有表
    tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    cupti_tables = [t for t in tables if 'CUPTI' in t.upper() or 'KERNEL' in t.upper()]
    print(f"Total tables: {len(tables)}")
    print(f"CUPTI/KERNEL tables: {cupti_tables}\n")

    # Runtime API 记录
    print("=== CUPTI_ACTIVITY_KIND_RUNTIME (first 5) ===")
    rows = conn.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME LIMIT 5').fetchall()
    for r in rows:
        print(dict(r))
    count = conn.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME').fetchone()[0]
    print(f"Total: {count}\n")

    # StringIds
    print("=== StringIds (first 30) ===")
    for r in conn.execute('SELECT * FROM StringIds LIMIT 30').fetchall():
        print(r)

    # Processes
    print("\n=== PROCESSES ===")
    for r in conn.execute('SELECT * FROM PROCESSES').fetchall():
        print(r)

    # NVTX
    try:
        print("\n=== NVTX_EVENTS (first 10) ===")
        for r in conn.execute('SELECT * FROM NVTX_EVENTS LIMIT 10').fetchall():
            print(dict(r))
        nvtx_count = conn.execute('SELECT COUNT(*) FROM NVTX_EVENTS').fetchone()[0]
        print(f"Total NVTX events: {nvtx_count}")
    except sqlite3.OperationalError:
        print("No NVTX_EVENTS table")

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <sqlite_path>")
        sys.exit(1)
    inspect(sys.argv[1])
