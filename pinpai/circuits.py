"""受限 R/C/L/CPE/Warburg 网络的表达式解析与阻抗计算。

表达式文法（大小写不敏感，序列化为小写规范形式）::

    top    := node ("," node)*          # 顶层逗号 = 串联
    node   := comb | element
    comb   := ("s"|"p") "(" [node ("," node)*] ")"
    element := ("R"|"C"|"L"|"W" | "CPE") [":" 标签]

* ``s(...)`` 串联，``p(...)`` 并联；同级同类在规范序列化时展平。
* 同类型元件省略标签时只能出现一次；带标签元件的参数键为 ``R:ct`` 这样的形式。
* 示例：``r,s(p(r:ct,cpe:dl),w)`` 表示 Rs 与（Rct//CPEdl）及 Warburg 串联。

阻抗约定（全文统一）
---------------------
* 时间因子约定为 :math:`e^{+j\\omega t}`，因此电容阻抗 ``Z_C = 1/(j w C)``，
  无源（passive）阻抗满足 ``Im(Z) < 0``。
* CPE：:math:`Z_{CPE}=1/[Q(j\\omega)^n]`，其中复幂取 **主值支**
  :math:`(j\\omega)^n=\\omega^n e^{+j n\\pi/2}`，``0 <= n <= 1``；
  ``n=1`` 即理想电容，``n`` 的物理上下界恒为 ``[0, 1]``。
* Warburg（半无限扩散）：:math:`Z_W=1/[W\\sqrt{j\\omega}]
  =(1-j)/(W\\sqrt{2\\omega})`，``W`` 为导纳系数（单位 ``S*s^0.5``）。
* Nyquist 图按 EIS 习惯以 ``-Im(Z)`` 为纵轴向上绘制。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

COMBINATORS = {"s", "p"}
ELEMENT_KINDS = {"R", "C", "L", "CPE", "W"}

#: 参数 -> （单位, 默认下界, 默认上界, 默认初值/数量级）
ELEMENT_META: Dict[str, Tuple[str, float, float, float]] = {
    "R": ("ohm", 1e-3, 1e9, 100.0),
    "C": ("F", 1e-15, 1.0, 1e-6),
    "L": ("H", 1e-12, 1e3, 1e-3),
    "Q": ("S*s^n", 1e-12, 1.0, 1e-6),
    "n": ("", 0.0, 1.0, 0.9),
    "W": ("S*s^0.5", 1e-9, 1e6, 1.0),
}
#: 每种元件使用的参数名（顺序固定）
ELEMENT_PARAMS: Dict[str, Tuple[str, ...]] = {
    "R": ("R",),
    "C": ("C",),
    "L": ("L",),
    "W": ("W",),
    "CPE": ("Q", "n"),
}
N_BOUNDS = (0.0, 1.0)
LABEL_RE = re.compile(r"^[A-Za-z0-9_]{1,24}$")


class CircuitError(ValueError):
    """表达式或参数定义不合法。"""


@dataclass
class Node:
    kind: str
    name: str | None = None
    children: List["Node"] = field(default_factory=list)

    def is_comb(self) -> bool:
        return self.kind in COMBINATORS


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
class _Parser:
    def __init__(self, text: str):
        self.t = text.strip()
        self.i = 0

    def _skip(self) -> None:
        while self.i < len(self.t) and self.t[self.i].isspace():
            self.i += 1

    def parse(self) -> Node:
        nodes = [self._node()]
        while True:
            self._skip()
            if self.i >= len(self.t):
                break
            if self.t[self.i] != ",":
                raise CircuitError(f"位置 {self.i}: 应为 ',' 或结束，得到 {self.t[self.i]!r}")
            self.i += 1
            nodes.append(self._node())
        if len(nodes) == 1:
            return nodes[0]
        return Node("s", children=nodes)

    def _node(self) -> Node:
        self._skip()
        start = self.i
        while self.i < len(self.t) and (self.t[self.i].isalnum() or self.t[self.i] == "_"):
            self.i += 1
        if self.i == start:
            got = self.t[self.i] if self.i < len(self.t) else "<结束>"
            raise CircuitError(f"位置 {start}: 缺少元件名，得到 {got!r}")
        ident = self.t[start:self.i].lower()
        self._skip()
        if ident in COMBINATORS:
            return self._comb(ident)
        if ident not in {"r", "c", "l", "cpe", "w"}:
            raise CircuitError(f"位置 {start}: 未知元件 {ident!r}（允许 R/C/L/CPE/W/s/p）")
        kind = ident.upper()
        name = None
        if self.i < len(self.t) and self.t[self.i] == ":":
            self.i += 1
            name = self._label()
        return Node(kind, name)

    def _comb(self, kind: str) -> Node:
        if self.i >= len(self.t) or self.t[self.i] != "(":
            raise CircuitError(f"位置 {self.i}: '{kind}' 后应为 '('")
        self.i += 1
        children: List[Node] = []
        self._skip()
        if self.i < len(self.t) and self.t[self.i] == ")":
            self.i += 1
            return Node(kind, children=children)
        while True:
            children.append(self._node())
            self._skip()
            if self.i < len(self.t) and self.t[self.i] == ",":
                self.i += 1
                continue
            if self.i < len(self.t) and self.t[self.i] == ")":
                self.i += 1
                break
            got = self.t[self.i] if self.i < len(self.t) else "<结束>"
            raise CircuitError(f"位置 {self.i}: '{kind}(...)' 内应为 ',' 或 ')'，得到 {got!r}")
        return Node(kind, children=children)

    def _label(self) -> str:
        self._skip()
        start = self.i
        while self.i < len(self.t) and (self.t[self.i].isalnum() or self.t[self.i] == "_"):
            self.i += 1
        label = self.t[start:self.i].lower()
        if not LABEL_RE.match(label):
            raise CircuitError(f"位置 {start}: 非法或缺失的元件标签 {label!r}")
        return label


def parse(text: str) -> Node:
    if not isinstance(text, str) or not text.strip():
        raise CircuitError("表达式为空")
    p = _Parser(text)
    node = p.parse()
    p._skip()
    if p.i != len(p.t):
        raise CircuitError(f"位置 {p.i}: 多余字符 {p.t[p.i:]!r}")
    _validate(node)
    return node


def canonical(node: Node) -> str:
    """规范（稳定）序列化：小写、展平同级同类组合、无空白。"""
    if node.is_comb():
        parts: List[str] = []
        for ch in node.children:
            sub = canonical(ch)
            if ch.is_comb() and ch.kind == node.kind:
                inner = sub[2:-1]  # 去掉 "s(" 与 ")"
                parts.append(inner)
            else:
                parts.append(sub)
        return f"{node.kind}(" + ",".join(parts) + ")"
    kind = node.kind.lower()
    return f"{kind}:{node.name}" if node.name else kind


def parse_canonical(text: str) -> str:
    return canonical(parse(text))


def _validate(node: Node) -> None:
    seen: set[str] = set()

    def walk(nd: Node) -> None:
        if nd.is_comb():
            if not nd.children:
                raise CircuitError(f"{nd.kind}(...) 至少需要一个子元件")
            for ch in nd.children:
                walk(ch)
            return
        for pk in element_param_keys(nd):
            if pk in seen:
                raise CircuitError(f"元件参数 {pk} 在表达式中重复出现，请用标签区分同类元件")
            seen.add(pk)

    walk(node)


def element_param_keys(node: Node) -> List[str]:
    return [f"{pname}:{node.name}" if node.name else pname
            for pname in ELEMENT_PARAMS[node.kind]]


def collect_param_keys(node: Node) -> List[str]:
    keys: List[str] = []
    stack = [node]
    while stack:
        nd = stack.pop()
        if nd.is_comb():
            stack.extend(reversed(nd.children))
        else:
            keys.extend(element_param_keys(nd))
    return keys


# --------------------------------------------------------------------------- #
# 参数定义校验
# --------------------------------------------------------------------------- #
def validate_param_defs(param_defs: List[dict], node: Node) -> List[dict]:
    """校验参数定义列表，返回规范化副本。键：key, unit, lo, hi, value, group。"""
    needed = set(collect_param_keys(node))
    seen: set[str] = set()
    out: List[dict] = []
    groups: Dict[str, Tuple[str, Tuple[float, float]]] = {}
    for raw in param_defs:
        pd = dict(raw)
        key = str(pd.get("key", ""))
        if key not in needed:
            raise CircuitError(f"参数 {key} 未出现在表达式 {canonical(node)} 中")
        if key in seen:
            raise CircuitError(f"参数 {key} 重复定义")
        seen.add(key)
        base = key.split(":", 1)[0]
        unit, dlo, dhi, dval = ELEMENT_META[base]
        declared = str(pd.get("unit") or unit)
        if declared != unit:
            raise CircuitError(f"参数 {key} 的单位应为 {unit!r}，得到 {declared!r}")
        try:
            lo, hi, value = float(pd["lo"]), float(pd["hi"]), float(pd["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CircuitError(f"参数 {key} 的 lo/hi/value 必须是数值") from exc
        if not (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(value)):
            raise CircuitError(f"参数 {key} 的 lo/hi/value 必须有限")
        if not (lo < hi):
            raise CircuitError(f"参数 {key} 需要严格 lo < hi")
        if not (lo <= value <= hi):
            raise CircuitError(f"参数 {key} 的初值 {value:g} 超出 [{lo:g}, {hi:g}]")
        if base == "n" and not (N_BOUNDS[0] <= lo and hi <= N_BOUNDS[1]):
            raise CircuitError(f"CPE 指数 {key} 的边界必须包含在物理范围 [0, 1] 内")
        grp = pd.get("group") or None
        if grp is not None:
            grp = str(grp)
            if not grp:
                grp = None
        if grp:
            if grp in groups:
                u0, (glo, ghi) = groups[grp]
                if u0 != unit:
                    raise CircuitError(f"共享组 {grp} 混合了不同单位（{u0} 与 {unit}）")
                ilo, ihi = max(glo, lo), min(ghi, hi)
                if not (ilo < ihi):
                    raise CircuitError(f"共享组 {grp} 成员边界交集为空，无法共享")
                groups[grp] = (u0, (ilo, ihi))
            else:
                groups[grp] = (unit, (lo, hi))
        out.append({"key": key, "unit": unit, "lo": lo, "hi": hi,
                    "value": value, "group": grp})
    missing = needed - seen
    if missing:
        raise CircuitError(f"表达式需要但未定义的参数：{sorted(missing)}")
    order = {k: i for i, k in enumerate(collect_param_keys(node))}
    out.sort(key=lambda d: order[d["key"]])
    return out


def default_param_defs(node: Node) -> List[dict]:
    defs = []
    for key in collect_param_keys(node):
        base = key.split(":", 1)[0]
        unit, lo, hi, val = ELEMENT_META[base]
        defs.append({"key": key, "unit": unit, "lo": lo, "hi": hi,
                     "value": val, "group": None})
    return defs


# --------------------------------------------------------------------------- #
# 阻抗
# --------------------------------------------------------------------------- #
def impedance(node: Node, values: Dict[str, float], omega: np.ndarray) -> np.ndarray:
    """按表达式与参数值计算复阻抗；omega 为角频率（rad/s，须为正）。"""
    kind = node.kind
    if kind == "s":
        z = np.zeros_like(omega, dtype=complex)
        for ch in node.children:
            z = z + impedance(ch, values, omega)
        return z
    if kind == "p":
        y = np.zeros_like(omega, dtype=complex)
        for ch in node.children:
            y = y + 1.0 / impedance(ch, values, omega)
        return 1.0 / y

    def val(pname: str) -> float:
        return values[f"{pname}:{node.name}" if node.name else pname]

    w = omega
    if kind == "R":
        return np.full_like(w, val("R"), dtype=complex)
    if kind == "C":
        return 1.0 / (1j * w * val("C"))
    if kind == "L":
        return 1j * w * val("L")
    if kind == "W":
        # sqrt(jω) = sqrt(ω) * exp(jπ/4)
        return 1.0 / (val("W") * np.sqrt(w) * np.exp(1j * math.pi / 4))
    if kind == "CPE":
        q, n = val("Q"), val("n")
        # (jω)^n 主值支
        jwn = (w ** n) * np.exp(1j * n * math.pi / 2)
        return 1.0 / (q * jwn)
    raise CircuitError(f"未知节点类型 {kind}")


def model_impedance(expr: str, param_defs: List[dict],
                    freqs_hz: np.ndarray) -> np.ndarray:
    node = parse(expr)
    values = {pd["key"]: float(pd["value"]) for pd in pd}
    return impedance(node, values, 2.0 * math.pi * freqs_hz)
