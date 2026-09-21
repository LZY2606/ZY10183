"""元件阻抗与网络合并。

口径约定（明文化，界面与 README 同口径）：

* 时间约定为 **e^{+jωt}**，即 Z(ω) = R + jX，**电容的虚部为负**：
  Z_C = 1/(jωC) = -j/(ωC)，Nyquist 图按 -Im(Z) 对 Re(Z) 画在第一象限。
* CPE（常相位元件）阻抗 ``Z_Q = 1 / (Y0 * (jω)^n)``。
  复幂取主值支（principal branch）::

      log(jω) = ln(ω) + j·π/2,   ω > 0
      (jω)^n = ω^n · (cos(nπ/2) + j·sin(nπ/2))

  因此 Z_Q = 1/(Y0 ω^n) · (cos(nπ/2) - j·sin(nπ/2))，
  n∈(0,1) 时虚部为负，与电容一致；n=1 退化为电容，n=0 退化为电阻。
* Warburg（半无限扩散）取 ``n = 1/2`` 的同一主值支：
  Z_W = 1/(Y0·(jω)^0.5) = (1 - j)/(Y0·sqrt(2ω))。
  本工具的 Y0 单位为 S·s^0.5；注意有些文献写 Z_W = σ(1-j)/sqrt(ω)，
  二者换算 σ = 1/(Y0·sqrt(2))。
* 串联阻抗相加；并联导纳（1/Z）相加。
"""

from __future__ import annotations

import cmath
import math

import numpy as np

from .circuit import KIND_ELEMENT, KIND_PARALLEL, KIND_SERIES


def _element_z(kind: str, values: dict[str, float], omega: complex) -> complex:
    if kind == "R":
        return complex(values["R"], 0.0)
    if kind == "C":
        return 1.0 / (1j * omega * values["C"])
    if kind == "L":
        return 1j * omega * values["L"]
    if kind == "Q":
        y0 = values["Y0"]
        n = values["n"]
        # 主值支：ω>0 时 arg(jω)=π/2。cmath.exp 显式取支，避免分支歧义。
        pw = cmath.exp(n * (cmath.log(omega) + 1j * math.pi / 2.0))
        return 1.0 / (y0 * pw)
    if kind == "W":
        pw = cmath.exp(0.5 * (cmath.log(omega) + 1j * math.pi / 2.0))
        return 1.0 / (values["Y0"] * pw)
    raise ValueError(f"未知元件种类 {kind!r}")


def impedance(node: tuple, params: dict[str, float], freqs) -> np.ndarray:
    """对频率数组（Hz，必须为正）计算复阻抗，返回 complex124/128 ndarray。"""
    freqs = np.asarray(freqs, dtype=float)
    if np.any(freqs <= 0):
        raise ValueError("阻抗计算仅接受正频率；非正频率须先在数据层剔除并报行号")
    elem_values = _bind(node, params)
    out = np.empty(len(freqs), dtype=np.complex128)
    for i, f in enumerate(freqs):
        omega = complex(2.0 * math.pi * f, 0.0)
        out[i] = _eval(node, elem_values, omega)
    return out


def impedance_simple(node: tuple, params: dict[str, float], freq: float) -> complex:
    omega = complex(2.0 * math.pi * freq, 0.0)
    return _eval(node, _bind(node, params), omega)


def _bind(node: tuple, params: dict[str, float]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for _, label in _labels(node):
        table[label] = {
            key.split(".", 1)[1]: float(val)
            for key, val in params.items() if key.rsplit(".", 1)[0] == label
        }
    return table


def _labels(node: tuple) -> list[tuple[str, str]]:
    from .circuit import elements
    return elements(node)


def _eval(node: tuple, elem_values: dict[str, dict[str, float]], omega: complex) -> complex:
    kind = node[0]
    if kind == KIND_ELEMENT:
        _, ekind, label = node
        return _element_z(ekind, elem_values[label], omega)
    zs = [_eval(child, elem_values, omega) for child in node[1]]
    if kind == KIND_SERIES:
        z = 0.0j
        for zi in zs:
            z += zi
        return z
    # 并联：导纳相加，可容纳零/无穷阻抗的极端初值。
    y = 0.0j
    for zi in zs:
        y += 1.0 / zi if zi != 0 else complex(1e300, 0.0)
    return 1.0 / y if y != 0 else complex(1e300, 0.0)
