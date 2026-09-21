"""固定 fixture：真值电路合成数据（含零/负频率行）与近等价候选。

真值：Rs + (Rct // CPE(Q,n))，两时间尺度被 CPE 展平后，在有限频带内与
简单的 Rs + Rc//C 几乎重合。数据用固定随机种子加噪，保证可重放。
"""
from __future__ import annotations
import math
import numpy as np

from . import circuits

TRUE_EXPR = "s(r,p(r:ct,cpe:dl))"
TRUE_PARAMS = {"R": 10.0, "R:ct": 200.0, "Q:dl": 2.0e-5, "n:dl": 0.88}

F_MIN, F_MAX, N_FREQ = 1.0, 1.0e4, 25
SEED = 1186
NOISE_FRAC = 0.005  # 0.5% 比例噪声


def true_param_defs() -> list[dict]:
    return [
        {"key": "R", "unit": "ohm", "lo": 1e-3, "hi": 1e6, "value": 10.0, "group": None},
        {"key": "R:ct", "unit": "ohm", "lo": 1e-3, "hi": 1e6, "value": 100.0, "group": None},
        {"key": "Q:dl", "unit": "S*s^n", "lo": 1e-12, "hi": 1.0, "value": 1e-5, "group": None},
        {"key": "n:dl", "unit": "", "lo": 0.0, "hi": 1.0, "value": 0.9, "group": None},
    ]


def synthetic_dataset(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    freqs = np.geomspace(F_MIN, F_MAX, N_FREQ)
    omega = 2.0 * math.pi * freqs
    node = circuits.parse(TRUE_EXPR)
    z = circuits.impedance(node, TRUE_PARAMS, omega)
    scale = np.abs(z)
    noise = rng.normal(0.0, NOISE_FRAC, size=(2, freqs.size)) * np.tile(scale, (2, 1))
    return freqs, z + noise[0] + 1j * noise[1]


def dataset_csv() -> str:
    from .dataio import to_csv
    freqs, z = synthetic_dataset()
    lines = ["freq_hz,zreal,zimag"]
    # 在首尾插入零频与负频行，验证“不进入对数轴但报出行位置”
    lines.append("0,1,2")
    lines.append("-10,3,4")
    for i, f in enumerate(freqs):
        lines.append(f"{f:.10g},{z.real[i]:.10g},{z.imag[i]:.10g}")
    return "\n".join(lines) + "\n"


# 候选 A：Rs + (Rc // C) —— 在观测频带内与真值几乎重合
CANDIDATE_A_EXPR = "s(r,p(r:c,c))"
CANDIDATE_A_PARAMS = [
    {"key": "R", "unit": "ohm", "lo": 1e-3, "hi": 1e6, "value": 50.0, "group": None},
    {"key": "R:c", "unit": "ohm", "lo": 1e-3, "hi": 1e6, "value": 50.0, "group": None},
    {"key": "C", "unit": "F", "lo": 1e-15, "hi": 1.0, "value": 1e-7, "group": None},
]

# 候选 B：Rs + R//CPE，CPE 指数上界被人为压到 0.75 —— 真解约 0.88，
# n 只会贴上界，得不到内部收敛，是“贴界而非收敛”的验收候选。
CANDIDATE_B_EXPR = "s(r,p(r:c,cpe:dl))"
CANDIDATE_B_PARAMS = [
    {"key": "R", "unit": "ohm", "lo": 1e-3, "hi": 1e6, "value": 50.0, "group": None},
    {"key": "R:c", "unit": "ohm", "lo": 1e-3, "hi": 1e6, "value": 100.0, "group": None},
    {"key": "Q:dl", "unit": "S*s^n", "lo": 1e-12, "hi": 1.0, "value": 1e-5, "group": None},
    {"key": "n:dl", "unit": "", "lo": 0.2, "hi": 0.75, "value": 0.6, "group": None},
]


# 候选 C：Rs + Rb + (Rct//CPE) —— 总串联电阻 R=Rs+Rb 可辨识，
# 但 Rs 与 Rb 的分配在观测窗内近乎不可辨识：
# 设计矩阵条件数 ~1e12，属于“形式满秩、实际近秩亏”，标准误不给出。
CANDIDATE_C_EXPR = "s(r,r:b,p(r:ct,cpe:dl))"
CANDIDATE_C_PARAMS = [
    {"key": "R", "unit": "ohm", "lo": 1e-3, "hi": 1e6, "value": 5.0, "group": None},
    {"key": "R:b", "unit": "ohm", "lo": 1e-3, "hi": 1e6, "value": 5.0, "group": None},
    {"key": "R:ct", "unit": "ohm", "lo": 1e-3, "hi": 1e6, "value": 100.0, "group": None},
    {"key": "Q:dl", "unit": "S*s^n", "lo": 1e-12, "hi": 1.0, "value": 2e-5, "group": None},
    {"key": "n:dl", "unit": "", "lo": 0.2, "hi": 1.0, "value": 0.9, "group": None},
]


def seed_bundle() -> dict:
    """返回写入空库的固定种子内容。"""
    return {
        "dataset": {"name": "合成：Rs+(Rct//CPE), n=0.88",
                    "note": "固定种子 1186，0.5% 比例噪声；含零频/负频行",
                    "csv": dataset_csv()},
        "circuits": [
            {"name": "A: Rs+(Rc//C) 简化 Randles", "expr": CANDIDATE_A_EXPR,
             "param_defs": CANDIDATE_A_PARAMS,
             "note": "观测频带内与真值近似等价"},
            {"name": "B: Rs+(R//CPE), n 上界 0.75", "expr": CANDIDATE_B_EXPR,
             "param_defs": CANDIDATE_B_PARAMS,
             "note": "n 只会贴上界：贴界而非真正收敛"},
            {"name": "C: Rs+Rb+(Rct//CPE) 近奇异", "expr": CANDIDATE_C_EXPR,
             "param_defs": CANDIDATE_C_PARAMS,
             "note": "Rs/Rb 分配近不可辨识：设计矩阵近秩亏，不给标准误"},
        ],
    }
