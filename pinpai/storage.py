"""SQLite 存储：数据集、候选电路、拟合运行（父子版本树）与整库导出/导入。

导入策略为“按 ID 替换”：导出 JSON 保留原主键，清空后重新导入可以得到
逐字节等价的运行记录，便于复核与重放。
"""
from __future__ import annotations
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    note TEXT DEFAULT '',
    freqs TEXT NOT NULL,
    zreal TEXT NOT NULL,
    zimag TEXT NOT NULL,
    excluded TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS circuits (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    expr TEXT NOT NULL,
    param_defs TEXT NOT NULL,
    note TEXT DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    circuit_id INTEGER NOT NULL REFERENCES circuits(id),
    parent_id INTEGER REFERENCES runs(id),
    label TEXT DEFAULT '',
    weight_mode TEXT NOT NULL,
    gamma REAL NOT NULL DEFAULT 0,
    result TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    folder = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


# --------------------------------------------------------------------------- #
# datasets / circuits
# --------------------------------------------------------------------------- #
def create_dataset(conn, name: str, note: str, freqs, z,
                   excluded: Optional[list] = None) -> int:
    cur = conn.execute(
        "INSERT INTO datasets(name,note,freqs,zreal,zimag,excluded,created_at)"
        " VALUES(?,?,?,?,?,?,?)",
        (name, note,
         json.dumps([float(x) for x in freqs]),
         json.dumps([float(x) for x in z.real]),
         json.dumps([float(x) for x in z.imag]),
         json.dumps(excluded or []), time.time()))
    conn.commit()
    return int(cur.lastrowid)


def get_dataset(conn, dataset_id: int) -> Optional[dict]:
    row = conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    return _dataset_row(row) if row else None


def list_datasets(conn) -> List[dict]:
    return [_dataset_row(r) for r in conn.execute(
        "SELECT * FROM datasets ORDER BY id")]


def _dataset_row(row) -> dict:
    return {"id": row["id"], "name": row["name"], "note": row["note"],
            "freqs": json.loads(row["freqs"]),
            "zreal": json.loads(row["zreal"]),
            "zimag": json.loads(row["zimag"]),
            "excluded": json.loads(row["excluded"]),
            "created_at": row["created_at"]}


def create_circuit(conn, name: str, expr: str, param_defs: list,
                   note: str = "") -> int:
    cur = conn.execute(
        "INSERT INTO circuits(name,expr,param_defs,note,created_at)"
        " VALUES(?,?,?,?,?)",
        (name, expr, json.dumps(param_defs), note, time.time()))
    conn.commit()
    return int(cur.lastrowid)


def update_circuit(conn, circuit_id: int, name: str, expr: str,
                   param_defs: list, note: str) -> None:
    conn.execute("UPDATE circuits SET name=?,expr=?,param_defs=?,note=? WHERE id=?",
                 (name, expr, json.dumps(param_defs), note, circuit_id))
    conn.commit()


def get_circuit(conn, circuit_id: int) -> Optional[dict]:
    row = conn.execute("SELECT * FROM circuits WHERE id=?", (circuit_id,)).fetchone()
    return _circuit_row(row) if row else None


def list_circuits(conn) -> List[dict]:
    return [_circuit_row(r) for r in conn.execute(
        "SELECT * FROM circuits ORDER BY id")]


def _circuit_row(row) -> dict:
    return {"id": row["id"], "name": row["name"], "expr": row["expr"],
            "param_defs": json.loads(row["param_defs"]),
            "note": row["note"], "created_at": row["created_at"]}


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
def create_run(conn, dataset_id: int, circuit_id: int, weight_mode: str,
               gamma: float, result: dict, parent_id: Optional[int] = None,
               label: str = "") -> int:
    cur = conn.execute(
        "INSERT INTO runs(dataset_id,circuit_id,parent_id,label,weight_mode,gamma,"
        "result,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (dataset_id, circuit_id, parent_id, label, weight_mode, float(gamma),
         json.dumps(result), time.time()))
    conn.commit()
    return int(cur.lastrowid)


def get_run(conn, run_id: int) -> Optional[dict]:
    row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["result"] = json.loads(row["result"])
    return d


def list_runs(conn, dataset_id: Optional[int] = None,
              circuit_id: Optional[int] = None) -> List[dict]:
    q = "SELECT id,dataset_id,circuit_id,parent_id,label,weight_mode,gamma,created_at FROM runs"
    cond, args = [], []
    if dataset_id is not None:
        cond.append("dataset_id=?"); args.append(dataset_id)
    if circuit_id is not None:
        cond.append("circuit_id=?"); args.append(circuit_id)
    if cond:
        q += " WHERE " + " AND ".join(cond)
    q += " ORDER BY id"
    return [dict(r) for r in conn.execute(q, args)]


# --------------------------------------------------------------------------- #
# 整库导出 / 导入 / 清空
# --------------------------------------------------------------------------- #
def export_all(conn) -> dict:
    def rows(table: str) -> list:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY id")]
    return {"format": "pinpai-tai/v1", "exported_at": time.time(),
            "datasets": rows("datasets"), "circuits": rows("circuits"),
            "runs": rows("runs")}


def wipe(conn) -> None:
    conn.executescript(
        "DELETE FROM runs; DELETE FROM circuits; DELETE FROM datasets;")
    conn.commit()


def import_all(conn, bundle: dict) -> dict:
    if bundle.get("format") != "pinpai-tai/v1":
        raise ValueError("导入包 format 不是 pinpai-tai/v1")
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        wipe(conn)
        counts = {}
        for table in ("datasets", "circuits", "runs"):
            n = 0
            for row in bundle.get(table, []):
                cols = [c for c in row.keys() if c != "id"]
                conn.execute(
                    f"INSERT INTO {table}(id,{','.join(cols)}) "
                    f"VALUES({','.join(['?'] * (len(cols) + 1))})",
                    [row["id"]] + [_json_like(table, c, row[c]) for c in cols])
                n += 1
            counts[table] = n
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys=ON")
    return counts


def _json_like(table: str, col: str, value: Any):
    if table == "datasets" and col in ("freqs", "zreal", "zimag", "excluded"):
        return value if isinstance(value, str) else json.dumps(value)
    if table == "circuits" and col == "param_defs":
        return value if isinstance(value, str) else json.dumps(value)
    if table == "runs" and col == "result":
        return value if isinstance(value, str) else json.dumps(value)
    return value
