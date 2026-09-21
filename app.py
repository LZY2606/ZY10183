"""频弧辨证台 —— FastAPI 本地服务入口。

运行：.venv/bin/uvicorn app:app --host 127.0.0.1 --port 5523
"""

from __future__ import annotations

import json
import os
from typing import Any

import contextlib

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from eis.circuit import (
    CircuitError, ELEMENTS, canonical, default_parameter_table, elements,
    parse,
)
from eis.db import (
    connect, create_circuit, create_dataset, export_all, get_circuit,
    get_dataset, get_run, import_payload, list_circuits, list_datasets,
    list_runs, reset, run_tree, seed_if_empty,
)
from eis.dataio import parse_table
from eis.service import build_run_view, run_fit

DB_PATH = os.environ.get("EIS_DB", os.path.join("data", "eis.db"))
_conns = {}


@contextlib.asynccontextmanager
async def lifespan(app):
    seed_if_empty(db())
    yield


app = FastAPI(title="频弧辨证台", version="1.0", lifespan=lifespan)


def db():
    import threading
    tid = threading.get_ident()
    if tid not in _conns:
        _conns[tid] = connect(DB_PATH)
        seed_if_empty(_conns[tid])
    return _conns[tid]


@app.get("/api/health")
def health():
    return {"ok": True, "title": "频弧辨证台"}


# ---------------- 数据模型 ----------------

class DatasetIn(BaseModel):
    name: str = "未命名数据集"
    text: str
    note: str = ""


class CircuitIn(BaseModel):
    name: str
    text: str
    parameters: list[dict] | None = None
    note: str = ""


class CircuitPatch(BaseModel):
    name: str | None = None
    parameters: list[dict] | None = None
    note: str | None = None


class FitIn(BaseModel):
    circuit_id: str
    dataset_id: str
    weighting: str = "sigma"
    n_starts: int = Field(default=12, ge=1, le=40)
    note: str = ""


class ChildFitIn(FitIn):
    parent_id: str
    action: str = "fix"  # fix=固定参数重跑；reweight=改权重子版本
    fixed: dict[str, float] | None = None
    version_label: str | None = None


class ImportIn(BaseModel):
    payload: dict[str, Any]


@app.exception_handler(CircuitError)
def _circuit_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ValueError)
def _value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------------- 元件目录 / 约定 ----------------

@app.get("/api/catalog")
def catalog():
    return {
        "elements": {
            k: {"kind": v.kind, "cn": v.cn, "params": v.params,
                "units": v.units, "defaults": v.defaults,
                "lower": v.lower, "upper": v.upper}
            for k, v in ELEMENTS.items()},
        "conventions": {
            "time": "e^(+jωt)，Z=R+jX，电容/CPE 的虚部为负，Nyquist 画 -Im vs Re",
            "cpe": "Z_Q = 1/(Y0·(jω)^n)，(jω)^n 取主值支: ω^n(cos nπ/2 + j sin nπ/2)",
            "warburg": "Z_W = 1/(Y0·(jω)^0.5) = (1-j)/(Y0√(2ω))",
            "n_bounds": "CPE 指数 n 默认边界 [0,1]，可在参数表逐元件改写",
            "series_parallel": "+ 串联，| 并联，| 优先于 +，括号显式分组",
            "weightings": {
                "sigma": "(Z-Zobs)/σ，使用数据列 sigma_re/sigma_im",
                "unit": "等权（欧姆最小二乘）",
                "modulus": "σ_i=|Zobs_i| 幅模加权",
            },
            "nonpositive_freq": "f<=0 不入对数轴/拟合，原始物理行号在排除列表报出",
            "rank": "SVD 阈值 1e-8：秩缺时不输出标准误，只报相关与零空间",
            "bound_vs_converge": "贴上/下界与数值收敛分开报告: bound_only != converged",
        },
    }


# ---------------- 数据集 ----------------

@app.get("/api/datasets")
def api_datasets():
    return list_datasets(db())


@app.get("/api/datasets/{ds_id}")
def api_dataset(ds_id: str):
    d = get_dataset(db(), ds_id)
    if d is None:
        raise HTTPException(404, "数据集不存在")
    return d


@app.post("/api/datasets")
def api_dataset_create(body: DatasetIn):
    parsed = parse_table(body.text)
    if parsed["valid_count"] == 0:
        raise HTTPException(400, "没有任何有效频率点（f>0 且数值完整）")
    ds_id = create_dataset(db(), body.name, body.text, source="upload",
                           note=body.note)
    return {"id": ds_id, "valid": parsed["valid_count"],
            "excluded": parsed["excluded"]}


# ---------------- 候选电路 ----------------

@app.get("/api/circuits")
def api_circuits():
    return list_circuits(db())


@app.get("/api/circuits/{cid}")
def api_circuit(cid: str):
    d = get_circuit(db(), cid)
    if d is None:
        raise HTTPException(404, "候选电路不存在")
    return d


@app.post("/api/circuits")
def api_circuit_create(body: CircuitIn):
    node = parse(body.text)  # CircuitError -> 400
    params = body.parameters or default_parameter_table(node)
    labels = {label for _, label in elements(node)}
    names = {p["name"] for p in default_parameter_table(node)}
    for p in params:
        if p["name"] not in names:
            raise CircuitError(f"参数 {p['name']} 不属于电路 {body.text}")
        if float(p["upper"]) <= float(p["lower"]):
            raise CircuitError(f"参数 {p['name']} 上界必须大于下界")
    cid = create_circuit(db(), body.name, body.text, params, note=body.note)
    return {"id": cid, "canonical": canonical(node)}


@app.patch("/api/circuits/{cid}")
def api_circuit_patch(cid: str, body: CircuitPatch):
    existing = get_circuit(db(), cid)
    if existing is None:
        raise HTTPException(404, "候选电路不存在")
    if body.name is not None:
        db().execute("UPDATE circuits SET name=? WHERE id=?", (body.name, cid))
    if body.note is not None:
        db().execute("UPDATE circuits SET note=? WHERE id=?", (body.note, cid))
    if body.parameters is not None:
        valid_names = {p["name"] for p in existing["parameters"]}
        for p in body.parameters:
            if p["name"] not in valid_names:
                raise CircuitError(f"未知参数 {p['name']}")
            if float(p["upper"]) <= float(p["lower"]):
                raise CircuitError(f"参数 {p['name']} 上界必须大于下界")
        db().execute("DELETE FROM circuit_params WHERE circuit_id=?", (cid,))
        for i, p in enumerate(body.parameters):
            db().execute(
                "INSERT INTO circuit_params(circuit_id,name,element,param,kind,"
                "unit,value,lower,upper,fixed,share,sort) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (cid, p["name"], p["element"], p["param"], p["kind"],
                 p.get("unit", ""), float(p["value"]), float(p["lower"]),
                 float(p["upper"]), 1 if p.get("fixed") else 0,
                 p.get("share", "") or "", i))
    db().commit()
    return {"ok": True}


@app.post("/api/circuits/preview")
def api_circuit_preview(body: CircuitIn):
    node = parse(body.text)
    return {"text": body.text, "canonical": canonical(node),
            "parameters": default_parameter_table(node)}


# ---------------- 拟合与版本 ----------------

@app.get("/api/runs")
def api_runs(dataset_id: str | None = None, circuit_id: str | None = None):
    return list_runs(db(), dataset_id, circuit_id)


@app.get("/api/runs/tree")
def api_run_tree():
    return run_tree(db())


@app.get("/api/runs/{rid}")
def api_run_detail(rid: str):
    row = get_run(db(), rid)
    if row is None:
        raise HTTPException(404, "运行不存在")
    return build_run_view(db(), row)


@app.post("/api/fit")
def api_fit(body: FitIn):
    rid, result, _, _ = run_fit(
        db(), circuit_id=body.circuit_id, dataset_id=body.dataset_id,
        weighting=body.weighting, n_starts=body.n_starts, note=body.note,
        version_label="v1")
    return {"id": rid, "status": result.status, "chi2": result.chi2,
            "near_singular": result.near_singular,
            "at_upper": result.at_upper, "at_lower": result.at_lower}


@app.post("/api/fit/child")
def api_fit_child(body: ChildFitIn):
    """固定参数或修改权重后重跑；父版本原样保留，新运行挂在 parent_id 下。"""
    parent = get_run(db(), body.parent_id)
    if parent is None:
        raise HTTPException(404, "父运行不存在")
    overrides: dict[str, dict] = {}
    fixed_note = ""
    label = body.version_label
    if body.action == "fix":
        fixed = body.fixed or {}
        if not fixed:
            raise HTTPException(400, "fix 动作需要 fixed 参数表")
        circuit = get_circuit(db(), parent["circuit_id"])
        current = {p["name"]: p["value"] for p in parent["params"]}
        for name, value in fixed.items():
            overrides[name] = {"fixed": True, "value": float(value)}
        fixed_note = "固定 " + ", ".join(
            f"{n}={v:g}" for n, v in fixed.items())
        label = label or f"固定 {len(fixed)} 项"
    elif body.action == "reweight":
        label = label or f"权重:{body.weighting}"
        fixed_note = f"权重改为 {body.weighting}（父 {body.parent_id}）"
    else:
        raise HTTPException(400, "action 仅支持 fix / reweight")
    rid, result, _, _ = run_fit(
        db(), circuit_id=parent["circuit_id"],
        dataset_id=parent["dataset_id"], weighting=body.weighting,
        n_starts=body.n_starts, note=body.note,
        parent_id=body.parent_id, version_label=label,
        overrides=overrides, fixed_note=fixed_note)
    return {"id": rid, "parent_id": body.parent_id, "status": result.status,
            "chi2": result.chi2, "near_singular": result.near_singular,
            "at_upper": result.at_upper, "at_lower": result.at_lower}


# ---------------- 导出 / 导入 / 清空 ----------------

@app.get("/api/export")
def api_export():
    return export_all(db())


@app.get("/api/export/download")
def api_export_file():
    import tempfile
    payload = export_all(db())
    path = os.path.join(tempfile.gettempdir(), "pinghu_seis_export.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return FileResponse(path, filename="pinghu_seis_export.json",
                        media_type="application/json")


@app.post("/api/import")
def api_import(body: ImportIn):
    counts = import_payload(db(), body.payload)
    return {"ok": True, "counts": counts}


@app.post("/api/reset")
def api_reset():
    reset(db(), reseed=True)
    return {"ok": True}


# ---------------- 页面 ----------------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join("static", "index.html"))
