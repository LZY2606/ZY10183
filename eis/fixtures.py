"""固定 fixture 生成：种子写死，结果可复现。

真值电路（C0）：Rs + (Rct || Cdl)，Randles 型 RC 并联串联欧姆电阻。
在 1 Hz–100 kHz 的观测频带内，再给一个 R-CPE 候选，其 Nyquist 弧
与纯 RC 候选几乎重合（n 贴着 1 的边界）；第三个候选含两个串联电阻
Rct + Rleak，在有限频带内只有二者之和可辨识 -> 设计矩阵近奇异。
"""

from __future__ import annotations

import os

import numpy as np

from .circuit import canonical, default_parameter_table, parse
from .dataio import DATA_HEADERS
from .impedance import impedance

SEED = 20240922
TRUE_CIRCUIT = "Rs+(Rct|Cdl)"
TRUE_VALUES = {"Rs.R": 10.0, "Rct.R": 100.0, "Cdl.C": 1e-4}
F_MIN, F_MAX, N_FREQ = 1.0, 1.0e5, 41


def _sigmas(freqs):
    """每点噪声尺度：真值 |Z| 的 0.5%，下限 0.1Ω（Re/Im 共用）。"""
    z = impedance(parse(TRUE_CIRCUIT), TRUE_VALUES, freqs)
    return np.maximum(0.005 * np.abs(z), 0.1)


def generate_fixture_text(seed: int = SEED) -> str:
    
    rng = np.random.default_rng(seed)
    freqs = np.geomspace(F_MIN, F_MAX, N_FREQ)
    z = impedance(parse(TRUE_CIRCUIT), TRUE_VALUES, freqs)
    s = _sigmas(freqs)
    z_noisy = z + rng.normal(0.0, 1.0, z.shape) * s + \
        1j * rng.normal(0.0, 1.0, z.shape) * s
    lines = [",".join(DATA_HEADERS), "# 真值: Rs=10Ω, Rct=100Ω, Cdl=100µF, 噪声 σ≈0.5%"]
    pairs = []
    for i, f in enumerate(freqs):
        pairs.append((f, z_noisy.real[i], z_noisy.imag[i], s[i], s[i]))
    # 两条不进入对数轴的脏数据，保留原始物理行号。
    pairs.insert(7, (0.0, 100.0, -50.0, 1.0, 1.0))
    pairs.insert(25, (-12.0, 100.0, -50.0, 1.0, 1.0))
    for f, re, im, sre, sim in pairs:
        lines.append(f"{f:.6g},{re:.8g},{im:.8g},{sre:.6g},{sim:.6g}")
    return "\n".join(lines) + "\n"


def build_candidates() -> list[dict]:
    """三个候选：同构 RC、结构不同的 R-CPE、近奇异双串联电阻。"""
    specs = [
        {
            "name": "C1_RC：Rs+(Rct|Cdl)",
            "text": "Rs+(Rct|Cdl)",
        },
        {
            "name": "C2_RCPE：Rs+(Rct|Q1)",
            "text": "Rs+(Rct|Q1)",
            "n_upper": 0.999,
            "n_init": 0.97,
        },
        {
            "name": "C3_近奇异：Rs+Rleak+(Rct|Cdl)",
            "text": "Rs+Rleak+(Rct|Cdl)",
        },
    ]
    out = []
    for spec in specs:
        node = parse(spec["text"])
        table = default_parameter_table(node)
        for row in table:
            if spec.get("n_upper") and row["name"] == "Q1.n":
                row["upper"] = spec["n_upper"]
                row["value"] = spec["n_init"]
            if row["name"] == "Rleak.R":
                row["value"] = 5.0
        out.append({
            "name": spec["name"],
            "text": spec["text"],
            "canonical": canonical(node),
            "parameters": table,
        })
    return out


FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures", "randles_rc.csv")


def write_fixture_file(path: str = FIXTURE_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = generate_fixture_text()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path
