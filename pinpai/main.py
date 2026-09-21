from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import circuits, dataio, fixtures, fitting, storage

DB_PATH = os.environ.get(
    "PINPAI_DB", os.path.join(os.path.dirname(os.path.dirname(__file__)), "pinpai.db"))

app = FastAPI(title="频弧辨证台", version="1.0")
conn = storage.connect(DB_PATH)


def _seed(db) -> None:
    bundle = fixtures.seed_bundle()
    parsed = dataio.parse_table(bundle["dataset"]["csv"])
    storage.create_dataset(
        db, bundle["dataset"]["name"], bundle["dataset"]["note"],
        parsed["freqs"], parsed["z"], parsed["excluded"])
    for c in bundle["circuits"]:
        node = circuits.parse(c["expr"])
        defs = circuits.validate_param_defs(c["param_defs"], node)
        storage.create_circuit(db, c["name"], circuits.canonical(node), defs, c["note"])


if not storage.list_datasets(conn):
    _seed(conn)


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class ParamDefIn(BaseModel):
    key: str
    unit: Optional[str] = None
    lo: float
    hi: float
    value: float
    group: Optional[str] = None


class CircuitIn(BaseModel):
    name: str
    expr: str
    param_defs: List[ParamDefIn]
    note: str = ""


class DatasetIn(BaseModel):
    name: str
    note: str = ""
    csv_text: str


class FitIn(BaseModel):
    dataset_id: int
    circuit_id: int
    weight_mode: str = "modulus"
    gamma: float = 0.0
    n_starts: int = Field(default=8, ge=1, le=24)
    parent_id: Optional[int] = None
    label: str = ""
    seed: int = 1186
    max_iter: int = Field(default=200, ge=10, le=2000)


class RefitIn(BaseModel):
    parent_id: int
    fixed: Dict[str, float] = {}           # 参数键 -> 固定值
    param_defs: Optional[List[ParamDefIn]] = None  # 覆盖后的参数定义（含缩界）
    weight_mode: Optional[str] = None
    gamma: Optional[float] = None
    n_starts: int = Field(default=8, ge=1, le=24)
    label: str = ""
    seed: int = 1186


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(
    os.path.dirname(__file__), "static")), name="static")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "title": "频弧辨证台"}


# --------------------------------------------------------------------------- #
# 校验工具
# --------------------------------------------------------------------------- #
@app.post("/api/validate/expr")
def validate_expr(body: Dict[str, Any]) -> dict:
    try:
        node = circuits.parse(body.get("expr", ""))
        keys = circuits.collect_param_keys(node)
        defs = None
        raw = body.get("param_defs")
        if raw:
            incoming = {d.get("key") for d in raw}
            # 只在参数集与表达式完全匹配时才校验数值/界，否则按新表达式补默认值
            if incoming == set(keys):
                defs = circuits.validate_param_defs([dict(d) for d in raw], node)
            else:
                old = {d.get("key"): dict(d) for d in raw}
                defs = []
                for d in circuits.default_param_defs(node):
                    if d["key"] in old:
                        d.update({k: old[d["key"]][k] for k in
                                  ("lo", "hi", "value", "group")
                                  if k in old[d["key"]]})
                    defs.append(d)
                defs = circuits.validate_param_defs(defs, node)
        return {"canonical": circuits.canonical(node),
                "param_keys": keys,
                "param_defs": defs or circuits.default_param_defs(node)}
    except circuits.CircuitError as exc:
        raise HTTPException(400, str(exc)) from exc


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
@app.get("/api/datasets")
def get_datasets() -> list:
    return storage.list_datasets(conn)


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: int) -> dict:
    ds = storage.get_dataset(conn, dataset_id)
    if not ds:
        raise HTTPException(404, "数据集不存在")
    return ds


@app.post("/api/datasets")
def post_dataset(body: DatasetIn) -> dict:
    try:
        parsed = dataio.parse_table(body.csv_text)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    did = storage.create_dataset(conn, body.name, body.note,
                                 parsed["freqs"], parsed["z"], parsed["excluded"])
    return {"id": did, "excluded": parsed["excluded"]}


# --------------------------------------------------------------------------- #
# 电路
# --------------------------------------------------------------------------- #
@app.get("/api/circuits")
def get_circuits() -> list:
    return storage.list_circuits(conn)


@app.post("/api/circuits")
def post_circuit(body: CircuitIn) -> dict:
    try:
        node = circuits.parse(body.expr)
        defs = circuits.validate_param_defs(
            [d.model_dump() for d in body.param_defs], node)
    except circuits.CircuitError as exc:
        raise HTTPException(400, str(exc)) from exc
    cid = storage.create_circuit(conn, body.name, circuits.canonical(node),
                                 defs, body.note)
    return {"id": cid, "canonical": circuits.canonical(node)}


@app.put("/api/circuits/{circuit_id}")
def put_circuit(circuit_id: int, body: CircuitIn) -> dict:
    if not storage.get_circuit(conn, circuit_id):
        raise HTTPException(404, "电路不存在")
    try:
        node = circuits.parse(body.expr)
        defs = circuits.validate_param_defs(
            [d.model_dump() for d in body.param_defs], node)
    except circuits.CircuitError as exc:
        raise HTTPException(400, str(exc)) from exc
    storage.update_circuit(conn, circuit_id, body.name,
                           circuits.canonical(node), defs, body.note)
    return {"id": circuit_id, "canonical": circuits.canonical(node)}


# --------------------------------------------------------------------------- #
# 拟合
# --------------------------------------------------------------------------- #
def _arrays(dataset_id: int):
    ds = storage.get_dataset(conn, dataset_id)
    if not ds:
        raise HTTPException(404, "数据集不存在")
    f = np.array(ds["freqs"], dtype=float)
    z = np.array(ds["zreal"], dtype=float) + 1j * np.array(ds["zimag"], dtype=float)
    return ds, f, z


def _do_fit(dataset_id: int, circuit_id: int, param_defs, weight_mode, gamma,
            n_starts, seed, parent_id, label, extra=None) -> dict:
    _, f, z = _arrays(dataset_id)
    result = fitting.run_fit(
        storage.get_circuit(conn, circuit_id)["expr"], param_defs, f, z,
        weight_mode=weight_mode, gamma=gamma, n_starts=n_starts, seed=seed)
    if extra:
        result.update(extra)
    run_id = storage.create_run(conn, dataset_id, circuit_id, weight_mode,
                                gamma, result, parent_id, label)
    return {"id": run_id, "result": result}


@app.post("/api/fit")
def fit(body: FitIn) -> dict:
    circ = storage.get_circuit(conn, body.circuit_id)
    if not circ:
        raise HTTPException(404, "电路不存在")
    return _do_fit(body.dataset_id, body.circuit_id, circ["param_defs"],
                   body.weight_mode, body.gamma, body.n_starts, body.seed,
                   body.parent_id, body.label)


@app.post("/api/refit")
def refit(body: RefitIn) -> dict:
    parent = storage.get_run(conn, body.parent_id)
    if not parent:
        raise HTTPException(404, "父运行不存在")
    circ = storage.get_circuit(conn, parent["circuit_id"])
    if body.param_defs is not None:
        defs = [d.model_dump() for d in body.param_defs]
    else:
        defs = [dict(d) for d in circ["param_defs"]]
    # 固定参数：把界收缩为单点附近（LM 内部变量因此锁死）。
    # 正参数对称取相对 epsilon，避免界越零；n 用绝对 epsilon 且不越 [0,1]。
    for pd in defs:
        if pd["key"] in body.fixed:
            v = float(body.fixed[pd["key"]])
            base = pd["key"].split(":", 1)[0]
            if not (pd["lo"] <= v <= pd["hi"]):
                raise HTTPException(400, f"固定值 {v:g} 超出 {pd['key']} 的界")
            if base in fitting.LOG_BASES:
                eps = max(abs(v), 1e-30) * 1e-8
                pd["lo"], pd["hi"] = v / (1 + 1e-8), v * (1 + 1e-8)
            else:
                pd["lo"], pd["hi"] = max(v - 1e-9, 0.0), min(v + 1e-9, 1.0)
            pd["value"] = v
    weight_mode = body.weight_mode or parent["weight_mode"]
    gamma = body.gamma if body.gamma is not None else parent["gamma"]
    return _do_fit(parent["dataset_id"], parent["circuit_id"], defs,
                   weight_mode, gamma, body.n_starts, body.seed,
                   body.parent_id, body.label,
                   extra={"fixed": body.fixed, "fitted_param_defs": defs})


@app.get("/api/runs")
def runs(dataset_id: Optional[int] = None, circuit_id: Optional[int] = None) -> list:
    return storage.list_runs(conn, dataset_id, circuit_id)


@app.get("/api/runs/{run_id}")
def run_detail(run_id: int) -> dict:
    r = storage.get_run(conn, run_id)
    if not r:
        raise HTTPException(404, "运行不存在")
    return r


# --------------------------------------------------------------------------- #
# 导出 / 导入
# --------------------------------------------------------------------------- #
@app.get("/api/export")
def export_all() -> JSONResponse:
    return JSONResponse(storage.export_all(conn))


@app.post("/api/import")
def import_all(bundle: Dict[str, Any]) -> dict:
    try:
        counts = storage.import_all(conn, bundle)
    except (ValueError, Exception) as exc:  # 主键/格式错误统一 400
        raise HTTPException(400, f"导入失败：{exc}") from exc
    return {"imported": counts}


@app.post("/api/reset")
def reset() -> dict:
    storage.wipe(conn)
    _seed(conn)
    return {"status": "reseeded"}
