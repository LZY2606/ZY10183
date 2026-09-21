"""拟合编排：取数/取电路 -> 构造问题 -> 求解 -> 落库 -> 组装视图数据。"""

from __future__ import annotations

import numpy as np

from .circuit import parse
from .db import save_run, valid_arrays
from .fitting import FitProblem, solve


def _normalize_params(rows, overrides):
    """overrides: {name: {value?, lower?, upper?, fixed?, share?}}。"""
    out = []
    for r in rows:
        p = {
            "name": r["name"], "element": r["element"], "param": r["param"],
            "kind": r["kind"], "unit": r.get("unit", ""),
            "value": float(r["value"]), "lower": float(r["lower"]),
            "upper": float(r["upper"]), "fixed": bool(r["fixed"]),
            "share": r.get("share", "") or "",
        }
        ov = overrides.get(p["name"]) if overrides else None
        if ov:
            for key in ("value", "lower", "upper"):
                if key in ov and ov[key] is not None:
                    p[key] = float(ov[key])
            if "fixed" in ov and ov["fixed"] is not None:
                p["fixed"] = bool(ov["fixed"])
            if "share" in ov and ov["share"] is not None:
                p["share"] = str(ov["share"]).strip()
        if p["upper"] <= p["lower"]:
            raise ValueError(f"参数 {p['name']} 的上界必须严格大于下界")
        if not (p["lower"] < p["value"] < p["upper"]) and not p["fixed"]:
            p["value"] = min(p["upper"] * 0.5, max(p["lower"] * 1.5,
                           p["lower"] + (p["upper"] - p["lower"]) * 0.5))
        out.append(p)
    return out


def run_fit(conn, *, circuit_id, dataset_id, weighting="unit", n_starts=12,
            parent_id=None, version_label=None, overrides=None, note="",
            fixed_note="", seed=20240922):
    from .db import get_circuit
    circuit = get_circuit(conn, circuit_id)
    if circuit is None:
        raise ValueError("候选电路不存在")
    rows, freqs, zobs, sr, si = valid_arrays(conn, dataset_id)
    if len(rows) < 2:
        raise ValueError("有效频率点不足，无法拟合")
    params = _normalize_params(circuit["parameters"], overrides)
    node = parse(circuit["text"])
    problem = FitProblem(node, params, freqs, zobs, sr, si, weighting)
    result = solve(problem, n_starts=n_starts, seed=seed)
    rid = save_run(
        conn, circuit_id, dataset_id, result, problem,
        parent_id=parent_id, version_label=version_label,
        weighting=weighting, n_starts=n_starts, note=note,
        fixed_note=fixed_note)
    return rid, result, problem, rows


def build_run_view(conn, run_row) -> dict:
    """根据落库的拟合值重算模型曲线，组装前端所需全部数据。"""
    from .db import get_circuit, get_dataset
    circuit = get_circuit(conn, run_row["circuit_id"])
    dataset = get_dataset(conn, run_row["dataset_id"])
    rows, freqs, zobs, sr, si = valid_arrays(conn, run_row["dataset_id"])
    from .db import get_run
    d = get_run(conn, run_row["id"]) if "params" not in run_row else run_row
    if "params" not in d:
        d = run_row
    values = {p["name"]: p["value"] for p in d["params"]}
    from .impedance import impedance
    zm = impedance(parse(circuit["text"]), values, freqs)

    def sigmas():
        if d["weighting"] == "sigma":
            return sr, si
        if d["weighting"] == "modulus":
            s = np.maximum(np.abs(zobs), 1e-30)
            return s, s
        return np.ones_like(freqs), np.ones_like(freqs)

    wsr, wsi = sigmas()
    line_map = {r["freq"]: r["line_no"] for r in rows}
    pts = []
    for i, f in enumerate(freqs):
        mag_o, mag_m = abs(zobs[i]), abs(zm[i])
        ph_o = np.degrees(np.angle(zobs[i]))
        ph_m = np.degrees(np.angle(zm[i]))
        pts.append({
            "freq": float(f), "line_no": line_map.get(f),
            "re_o": float(zobs[i].real), "im_o": float(zobs[i].imag),
            "re_m": float(zm[i].real), "im_m": float(zm[i].imag),
            "mag_o": float(mag_o), "mag_m": float(mag_m),
            "phase_o": float(ph_o), "phase_m": float(ph_m),
            "res_re": float((zm[i].real - zobs[i].real) / wsr[i]),
            "res_im": float((zm[i].imag - zobs[i].imag) / wsi[i]),
        })
    return {
        "run": d, "circuit_name": circuit["name"],
        "circuit_text": circuit["text"], "canonical": circuit["canonical"],
        "dataset_name": dataset["name"], "points": pts,
        "excluded": [{"line_no": p["line_no"], "row_num": p["row_num"],
                      "freq": p["freq"], "reason": p["reason"]}
                     for p in dataset["points"] if p["status"] != "ok"],
    }
