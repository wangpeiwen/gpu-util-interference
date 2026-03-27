"""
MLWD SQLite 数据库 CRUD 操作。
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any

from .mlwd_schema import SCHEMA_SQL, DeploymentConfig, OperatorProfile


class MLWDDatabase:
    def __init__(self, db_path: str = "mlwd_data.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ── DeploymentConfig ──

    def upsert_config(self, cfg: DeploymentConfig) -> int:
        """插入或获取已有的部署配置，返回 config_id。"""
        row = self.conn.execute(
            "SELECT config_id FROM deployment_configs "
            "WHERE model=? AND framework=? AND quantization=? AND tp_degree=?",
            (cfg.model, cfg.framework, cfg.quantization, cfg.tp_degree),
        ).fetchone()
        if row:
            return row["config_id"]

        cur = self.conn.execute(
            "INSERT INTO deployment_configs (model, framework, quantization, tp_degree, gpu_model, cuda_version) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cfg.model, cfg.framework, cfg.quantization, cfg.tp_degree, cfg.gpu_model, cfg.cuda_version),
        )
        self.conn.commit()
        return cur.lastrowid

    # ── OperatorProfile ──

    def upsert_profile(self, profile: OperatorProfile):
        """插入或更新算子画像记录（UPSERT）。"""
        self.conn.execute(
            """INSERT INTO mlwd_operator_profiles
               (config_id, batch_size, seq_len, phase,
                ci_attn, ci_ffn, l2_attn, l2_ffn,
                sigma_bs, sigma_cu, sigma_l2, sigma_bw,
                t_attn, t_ffn, g_launch, r_attn, r_ffn, f_switch, ipc,
                t_attn_std, t_ffn_std, baseline_latency_ms)
               VALUES (?,?,?,?, ?,?,?,?, ?,?,?,?, ?,?,?,?,?,?,?, ?,?,?)
               ON CONFLICT(config_id, batch_size, seq_len, phase)
               DO UPDATE SET
                ci_attn=COALESCE(excluded.ci_attn, ci_attn),
                ci_ffn=COALESCE(excluded.ci_ffn, ci_ffn),
                l2_attn=COALESCE(excluded.l2_attn, l2_attn),
                l2_ffn=COALESCE(excluded.l2_ffn, l2_ffn),
                sigma_bs=COALESCE(excluded.sigma_bs, sigma_bs),
                sigma_cu=COALESCE(excluded.sigma_cu, sigma_cu),
                sigma_l2=COALESCE(excluded.sigma_l2, sigma_l2),
                sigma_bw=COALESCE(excluded.sigma_bw, sigma_bw),
                t_attn=COALESCE(excluded.t_attn, t_attn),
                t_ffn=COALESCE(excluded.t_ffn, t_ffn),
                g_launch=COALESCE(excluded.g_launch, g_launch),
                r_attn=COALESCE(excluded.r_attn, r_attn),
                r_ffn=COALESCE(excluded.r_ffn, r_ffn),
                f_switch=COALESCE(excluded.f_switch, f_switch),
                ipc=COALESCE(excluded.ipc, ipc),
                t_attn_std=COALESCE(excluded.t_attn_std, t_attn_std),
                t_ffn_std=COALESCE(excluded.t_ffn_std, t_ffn_std),
                baseline_latency_ms=COALESCE(excluded.baseline_latency_ms, baseline_latency_ms)
            """,
            (profile.config_id, profile.batch_size, profile.seq_len, profile.phase,
             profile.ci_attn, profile.ci_ffn, profile.l2_attn, profile.l2_ffn,
             profile.sigma_bs, profile.sigma_cu, profile.sigma_l2, profile.sigma_bw,
             profile.t_attn, profile.t_ffn, profile.g_launch,
             profile.r_attn, profile.r_ffn, profile.f_switch, profile.ipc,
             profile.t_attn_std, profile.t_ffn_std, profile.baseline_latency_ms),
        )
        self.conn.commit()

    def get_profile(self, config_id: int, batch_size: int, seq_len: int, phase: str) -> Optional[OperatorProfile]:
        """精确查询单条记录。"""
        row = self.conn.execute(
            "SELECT * FROM mlwd_operator_profiles "
            "WHERE config_id=? AND batch_size=? AND seq_len=? AND phase=?",
            (config_id, batch_size, seq_len, phase),
        ).fetchone()
        if not row:
            return None
        return OperatorProfile(**{k: row[k] for k in row.keys()})

    def get_nearest_profile(self, config_id: int, batch_size: int, seq_len: int, phase: str) -> Optional[OperatorProfile]:
        """近邻检索：找 (b, s) 最接近的记录。"""
        row = self.conn.execute(
            """SELECT *, ABS(batch_size - ?) + ABS(seq_len - ?) AS dist
               FROM mlwd_operator_profiles
               WHERE config_id=? AND phase=?
               ORDER BY dist ASC LIMIT 1""",
            (batch_size, seq_len, config_id, phase),
        ).fetchone()
        if not row:
            return None
        d = {k: row[k] for k in row.keys() if k != "dist"}
        return OperatorProfile(**d)

    def list_profiles(self, config_id: int) -> List[OperatorProfile]:
        """列出某部署配置下的所有画像。"""
        rows = self.conn.execute(
            "SELECT * FROM mlwd_operator_profiles WHERE config_id=? ORDER BY phase, batch_size, seq_len",
            (config_id,),
        ).fetchall()
        return [OperatorProfile(**{k: r[k] for k in r.keys()}) for r in rows]

    def is_collected(self, config_id: int, batch_size: int, seq_len: int, phase: str, stage: str) -> bool:
        """检查某阶段是否已采集（用于断点续采）。"""
        profile = self.get_profile(config_id, batch_size, seq_len, phase)
        if profile is None:
            return False
        if stage == "ncu":
            return profile.ci_attn is not None
        elif stage == "nsys":
            return profile.t_attn is not None
        elif stage == "sensitivity":
            return profile.sigma_bs is not None
        return False

    def export_all(self) -> List[Dict[str, Any]]:
        """导出全部数据为字典列表。"""
        rows = self.conn.execute(
            """SELECT d.model, d.framework, d.quantization, d.tp_degree, d.gpu_model,
                      p.*
               FROM mlwd_operator_profiles p
               JOIN deployment_configs d ON p.config_id = d.config_id
               ORDER BY d.model, p.phase, p.batch_size, p.seq_len"""
        ).fetchall()
        return [dict(r) for r in rows]
