"""SQLite 持久层：数据集 / 候选电路 / 拟合运行（父子版本）。

导出为单个 JSON（含建表所需的全部行），导入时清空后原样写回，
可用于“清空数据库 -> 重新导入 -> 复核”的闭环。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid

from .circuit import canonical as canon_text
from .circuit import default_parameter_table, parse
from .fixtures import build_candidates, generate_fixture_text
from .dataio import parse_table

DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "eis.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY, name TEXT, created_at REAL,
    raw_text TEXT, note TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id TEXT, line_no INTEGER, row_num INTEGER,
    freq REAL, z_re REAL, z_im REAL, sigma_re REAL, sigma_im REAL,
    status TEXT, reason TEXT,
    FOREIGN KEY(dataset_id) REFERENCES datasets(id)
);
CREATE TABLE IF NOT EXISTS circuits (
    id TEXT PRIMARY KEY, name TEXT, text TEXT, canonical TEXT,
    created_at REAL, note TEXT
);
CREATE TABLE IF NOT EXISTS circuit_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    circuit_id TEXT, name TEXT, element TEXT, param TEXT, kind TEXT,
    unit TEXT, value REAL, lower REAL, upper REAL, fixed INTEGER, share TEXT,
    sort INTEGER,
    FOREIGN KEY(circuit_id) REFERENCES circuits(id)
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, circuit_id TEXT, dataset_id TEXT,
    parent_id TEXT, version_label TEXT, created_at REAL,
    weighting TEXT, n_starts INTEGER, note TEXT,
    chi2 REAL, redchi2 REAL, dof INTEGER, rms REAL, status TEXT,
    iterations INTEGER, rank INTEGER, nvar INTEGER,
    condition REAL, near_singular INTEGER,
    at_lower TEXT, at_upper TEXT, null_directions TEXT,
    fixed_note TEXT
);
CREATE TABLE IF NOT EXISTS run_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT, name TEXT, value REAL, se REAL, fixed INTEGER,
    share TEXT, unit TEXT, variable_index INTEGER
);
CREATE TABLE IF NOT EXISTS run_starts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT, idx INTEGER, chi2 REAL, status TEXT, iterations INTEGER,
    at_lower TEXT, at_upper TEXT, x TEXT
);
CREATE TABLE IF NOT EXISTS run_corr (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT, a TEXT, b TEXT, value REAL
);
"""

TABLES = ("datasets", "points", "circuits", "circuit_params", "runs",
          "run_params", "run_starts", "run_corr", "meta")


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    if db_path != ":memory:":
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------- 数据集 ----------------

def create_dataset(conn, name, raw_text, source="upload", note=""):
    parsed = parse_table(raw_text)
    ds_id = new_id("ds")
    conn.execute(
        "INSERT INTO datasets(id,name,created_at,raw_text,note,source) "
        "VALUES(?,?,?,?,?,?)",
        (ds_id, name, time.time(), raw_text, note, source))
    for r in parsed["rows"]:
        conn.execute(
            "INSERT INTO points(dataset_id,line_no,row_num,freq,z_re,z_im,"
            "sigma_re,sigma_im,status,reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (ds_id, r["line"], r["rownum"], r["freq"], r["z_re"], r["z_im"],
             r["sigma_re"], r["sigma_im"], r["status"], r["reason"]))
    conn.commit()
    return ds_id


def get_dataset(conn, ds_id):
    row = conn.execute("SELECT * FROM datasets WHERE id=?", (ds_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["points"] = [dict(r) for r in conn.execute(
        "SELECT * FROM points WHERE dataset_id=? ORDER BY id", (ds_id,))]
    return d


def list_datasets(conn):
    rows = conn.execute(
        "SELECT d.*, SUM(CASE WHEN p.status='ok' THEN 1 ELSE 0 END) AS n_ok,"
        " SUM(CASE WHEN p.status!='ok' THEN 1 ELSE 0 END) AS n_excluded "
        "FROM datasets d LEFT JOIN points p ON p.dataset_id=d.id "
        "GROUP BY d.id ORDER BY d.created_at").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        out.append(d)
    return out


def valid_arrays(conn, ds_id):
    import numpy as np
    rows = conn.execute(
        "SELECT * FROM points WHERE dataset_id=? AND status='ok' "
        "ORDER BY freq", (ds_id,)).fetchall()
    f = np.array([r["freq"] for r in rows], dtype=float)
    z = np.array([complex(r["z_re"], r["z_im"]) for r in rows], dtype=complex)
    sr = np.array([r["sigma_re"] for r in rows], dtype=float)
    si = np.array([r["sigma_im"] for r in rows], dtype=float)
    return rows, f, z, sr, si


# ---------------- 候选电路 ----------------

def create_circuit(conn, name, text, parameters=None, note=""):
    node = parse(text)
    cid = new_id("c")
    if parameters is None:
        parameters = default_parameter_table(node)
    conn.execute(
        "INSERT INTO circuits(id,name,text,canonical,created_at,note) "
        "VALUES(?,?,?,?,?,?)",
        (cid, name, text, canon_text(node), time.time(), note))
    for i, p in enumerate(parameters):
        conn.execute(
            "INSERT INTO circuit_params(circuit_id,name,element,param,kind,"
            "unit,value,lower,upper,fixed,share,sort) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, p["name"], p["element"], p["param"], p["kind"],
             p.get("unit", ""), float(p["value"]), float(p["lower"]),
             float(p["upper"]), 1 if p.get("fixed") else 0,
             p.get("share", ""), i))
    conn.commit()
    return cid


def _params_of(conn, cid):
    rows = conn.execute(
        "SELECT * FROM circuit_params WHERE circuit_id=? ORDER BY sort",
        (cid,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["fixed"] = bool(d["fixed"])
        out.append(d)
    return out


def get_circuit(conn, cid):
    row = conn.execute("SELECT * FROM circuits WHERE id=?", (cid,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["parameters"] = _params_of(conn, cid)
    return d


def list_circuits(conn):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM circuits ORDER BY created_at")]


# ---------------- 拟合运行 ----------------

def save_run(conn, circuit_id, dataset_id, fit_result, problem, *,
             parent_id=None, version_label=None, weighting=None,
             n_starts=None, note="", fixed_note=""):
    rid = new_id("run")
    var_index = {}
    for k, var in enumerate(problem.variables):
        for nm in var["names"]:
            var_index[nm] = k
    conn.execute(
        "INSERT INTO runs(id,circuit_id,dataset_id,parent_id,version_label,"
        "created_at,weighting,n_starts,note,chi2,redchi2,dof,rms,status,"
        "iterations,rank,nvar,condition,near_singular,at_lower,at_upper,"
        "null_directions,fixed_note) VALUES("
        "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (rid, circuit_id, dataset_id, parent_id, version_label, time.time(),
         weighting or fit_result.weighting, n_starts, note,
         fit_result.chi2, fit_result.redchi2, fit_result.dof, fit_result.rms,
         fit_result.status, fit_result.iterations, fit_result.rank,
         fit_result.nvar, fit_result.condition,
         1 if fit_result.near_singular else 0,
         json.dumps(fit_result.at_lower), json.dumps(fit_result.at_upper),
         json.dumps(fit_result.null_directions), fixed_note))
    fixed_names = {p["name"] for p in problem.parameters if p.get("fixed")}
    for p in problem.parameters:
        nm = p["name"]
        conn.execute(
            "INSERT INTO run_params(run_id,name,value,se,fixed,share,unit,"
            "variable_index) VALUES(?,?,?,?,?,?,?,?)",
            (rid, nm, fit_result.values[nm],
             None if fit_result.se.get(nm) is None else fit_result.se[nm],
             1 if nm in fixed_names else 0, p.get("share", ""),
             p.get("unit", ""), var_index.get(nm)))
    for s in fit_result.starts:
        conn.execute(
            "INSERT INTO run_starts(run_id,idx,chi2,status,iterations,"
            "at_lower,at_upper,x) VALUES(?,?,?,?,?,?,?,?)",
            (rid, s["index"], s["chi2"], s["status"], s["iterations"],
             json.dumps(s["at_lower"]), json.dumps(s["at_upper"]),
             json.dumps(s["x"])))
    for (a, b), val in fit_result.correlation.items():
        conn.execute(
            "INSERT INTO run_corr(run_id,a,b,value) VALUES(?,?,?,?)",
            (rid, a, b, val))
    conn.commit()
    return rid


def _run_row(conn, rid):
    return conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()


def list_runs(conn, dataset_id=None, circuit_id=None):
    q = "SELECT * FROM runs WHERE 1=1"
    args = []
    if dataset_id:
        q += " AND dataset_id=?"; args.append(dataset_id)
    if circuit_id:
        q += " AND circuit_id=?"; args.append(circuit_id)
    q += " ORDER BY created_at"
    return [dict(r) for r in conn.execute(q, args)]


def get_run(conn, rid):
    row = _run_row(conn, rid)
    if row is None:
        return None
    d = dict(row)
    for key in ("at_lower", "at_upper", "null_directions"):
        d[key] = json.loads(d[key])
    d["params"] = [dict(r) for r in conn.execute(
        "SELECT * FROM run_params WHERE run_id=? ORDER BY id", (rid,))]
    d["starts"] = []
    for r in conn.execute(
            "SELECT * FROM run_starts WHERE run_id=? ORDER BY idx", (rid,)):
        sd = dict(r)
        sd["at_lower"] = json.loads(sd["at_lower"])
        sd["at_upper"] = json.loads(sd["at_upper"])
        sd["x"] = json.loads(sd["x"])
        d["starts"].append(sd)
    d["corr"] = [dict(r) for r in conn.execute(
        "SELECT a,b,value FROM run_corr WHERE run_id=?", (rid,))]
    return d


def run_tree(conn):
    """返回按 dataset/circuit 分组的父子版本树。"""
    runs = list_runs(conn)
    by_parent = {}
    for r in runs:
        by_parent.setdefault(r["parent_id"], []).append(r["id"])
    return {"runs": runs, "children": by_parent}


# ---------------- 播种 / 导出 / 导入 / 清空 ----------------

def seed_if_empty(conn, force=False):
    n = conn.execute("SELECT COUNT(*) c FROM datasets").fetchone()["c"]
    if n and not force:
        return None
    if force:
        reset(conn, reseed=False)
    ds = create_dataset(
        conn, "固定 fixture：Randles RC（含 f<=0 脏行）",
        generate_fixture_text(), source="fixture",
        note="真值 Rs=10Ω, Rct=100Ω, Cdl=100µF；1–1e5 Hz；σ≈0.5%")
    for cand in build_candidates():
        create_circuit(conn, cand["name"], cand["text"], cand["parameters"],
                       note="固定 fixture 候选")
    conn.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES('seeded',?)",
        (str(time.time()),))
    conn.commit()
    return ds


def export_all(conn) -> dict:
    payload = {"format": "pingu-seis-1", "exported_at": time.time(), "tables": {}}
    for table in TABLES:
        payload["tables"][table] = [dict(r) for r in conn.execute(
            f"SELECT * FROM {table}")]
    return payload


def reset(conn, reseed=True):
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        for table in TABLES:
            conn.execute(f"DELETE FROM {table}")
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
    if reseed:
        seed_if_empty(conn)


def import_payload(conn, payload: dict) -> dict:
    """清空后按导出快照原样写回（保留主键，保证父子引用可追溯）。"""
    if payload.get("format") != "pingu-seis-1":
        raise ValueError("导出文件 format 标识不是 pingu-seis-1")
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        for table in TABLES:
            conn.execute(f"DELETE FROM {table}")
        counts = {}
        for table in TABLES:
            rows = payload.get("tables", {}).get(table, [])
            counts[table] = len(rows)
            if not rows:
                continue
            cols = list(rows[0].keys())
            placeholders = ",".join("?" for _ in cols)
            conn.executemany(
                f"INSERT INTO {table}({','.join(cols)}) VALUES({placeholders})",
                [[r.get(c) for c in cols] for r in rows])
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
    return counts
