"""阻抗数据导入：CSV/TSV 文本 -> 频率（Hz）与复阻抗。

数据口径
--------
* 支持列：``freq_hz, zreal, zimag``（必需），可带表头；也接受
  ``f, Re, Im`` 等常见别名。行号从 **1** 开始，带表头时数据首行为 2。
* 频率必须为正且有限才进入对数轴/拟合；``freq <= 0`` 或非数的行被剔除，
  但原始行号在返回值 ``excluded`` 中逐条报出。
* 阻抗分量须为有限数，否则该行同样剔除并报行号与原因。
"""
from __future__ import annotations
import csv
import io
import math
from typing import Dict, List, Tuple

import numpy as np

FREQ_ALIASES = {"freq_hz", "f", "freq", "frequency", "频率"}
RE_ALIASES = {"zreal", "re", "zre", "real", "z_real", "实部"}
IM_ALIASES = {"zimag", "im", "zim", "imag", "z_imag", "虚部"}


def parse_table(text: str) -> Dict[str, object]:
    """解析 CSV/TSV 文本。返回 {freqs, z, excluded, total_rows}。"""
    sample = text[:2048]
    try:
        dialect = csv.excel_tab if sample.count("\t") > sample.count(",") else csv.excel
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    rows = list(reader)
    if not rows:
        raise ValueError("文件为空")

    header = [c.strip().lower() for c in rows[0]]
    has_header = any(h in FREQ_ALIASES | RE_ALIASES | IM_ALIASES for h in header)

    def col_index(names: set[str], fallback: int) -> int:
        for j, h in enumerate(header):
            if h in names:
                return j
        return fallback

    if has_header:
        ifi = col_index(FREQ_ALIASES, 0)
        ire = col_index(RE_ALIASES, 1)
        iim = col_index(IM_ALIASES, 2)
        data_rows = rows[1:]
        first_lineno = 2
    else:
        ifi, ire, iim = 0, 1, 2
        data_rows = rows
        first_lineno = 1

    freqs: List[float] = []
    zreal: List[float] = []
    zimag: List[float] = []
    excluded: List[dict] = []
    for offset, row in enumerate(data_rows):
        lineno = first_lineno + offset
        if not row or all(not c.strip() for c in row):
            continue
        try:
            f = float(row[ifi]); re_ = float(row[ire]); im_ = float(row[iim])
        except (IndexError, ValueError):
            excluded.append({"line": lineno, "reason": "无法解析为三个数值"})
            continue
        if not math.isfinite(f):
            excluded.append({"line": lineno, "reason": "频率非有限值"})
            continue
        if f <= 0.0:
            excluded.append({"line": lineno, "reason": f"频率 {f:g} Hz 非正，不进入对数轴"})
            continue
        if not (math.isfinite(re_) and math.isfinite(im_)):
            excluded.append({"line": lineno, "reason": "阻抗实部或虚部非有限值"})
            continue
        freqs.append(f)
        zreal.append(re_)
        zimag.append(im_)

    if not freqs:
        raise ValueError("没有可用的正频率数据行")
    return {
        "freqs": np.array(freqs, dtype=float),
        "z": np.array(zreal, dtype=float) + 1j * np.array(zimag, dtype=float),
        "excluded": excluded,
        "total_rows": len(freqs) + len(excluded),
    }


def to_csv(freqs: np.ndarray, z: np.ndarray) -> str:
    buf = io.StringIO()
    buf.write("freq_hz,zreal,zimag\n")
    for f, zz in zip(freqs, z):
        buf.write(f"{f:.10g},{zz.real:.10g},{zz.imag:.10g}\n")
    return buf.getvalue()
