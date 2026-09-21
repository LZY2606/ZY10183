"""受限 R / C / L / CPE / Warburg 串并联网络的稳定表达。

电路文本（circuit text）文法（大小写不敏感）::

    series   := parallel ('+' parallel)*
    parallel := atom ('|' atom)*
    atom     := '(' series ')' | ELEMENT

即 ``+`` 表示串联、``|`` 表示并联，``|`` 的结合优先级高于 ``+``，
要表达 "(R1+R2) 与 C1 并联" 必须写 ``(R1+R2)|C1``。

元件名首字母决定元件种类，其余字符是实例标签：

* ``R`` 电阻；``C`` 电容；``L`` 电感；
* ``Q`` 常相位元件 CPE，参数为 ``Y0`` 与指数 ``n``；
* ``W`` 半无限扩散 Warburg，阻抗与 CPE 同形但指数固定为 0.5，
  只有一个自由参数 ``Y0``（见 :mod:`eis.impedance`）。

参数名一律按 ``<元件标签>.<参数名>`` 引用，例如 ``R1.R``、``Q1.n``。

节点为不可变元组：``('s', children)``、``('p', children)``、
``('e', kind, label)``。:func:`canonical` 对同级子网络做字典序排序，
因此同一拓扑无论书写顺序如何，稳定表达都相同；比较两个结构不同的
电路是否给出同一拓扑，只需比较稳定表达。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

KIND_SERIES = "s"
KIND_PARALLEL = "p"
KIND_ELEMENT = "e"

_ELEMENT_RE = re.compile(r"^([RCLQW])([A-Za-z0-9]*)$")
_TOKEN_RE = re.compile(r"\s*([()+|]|[RCLQW][A-Za-z0-9]*)\s*", re.IGNORECASE)


class CircuitError(ValueError):
    """电路文本非法或参数表与电路不一致时抛出。"""


@dataclass(frozen=True)
class ElementSpec:
    kind: str
    cn: str
    params: tuple[str, ...]
    units: dict[str, str]
    defaults: dict[str, float]
    lower: dict[str, float]
    upper: dict[str, float]


# 默认数量级口径（SI）：R=Ω, C=F, L=H, Y0=S·s^n, n 无量纲。
ELEMENTS: dict[str, ElementSpec] = {
    "R": ElementSpec(
        "R", "电阻", ("R",), {"R": "Ω"},
        {"R": 100.0}, {"R": 1e-6}, {"R": 1e6},
    ),
    "C": ElementSpec(
        "C", "电容", ("C",), {"C": "F"},
        {"C": 1e-6}, {"C": 1e-12}, {"C": 1e-2},
    ),
    "L": ElementSpec(
        "L", "电感", ("L",), {"L": "H"},
        {"L": 1e-6}, {"L": 1e-9}, {"L": 1e2},
    ),
    "Q": ElementSpec(
        "Q", "常相位元件 CPE", ("Y0", "n"),
        {"Y0": "S·s^n", "n": ""},
        {"Y0": 1e-6, "n": 0.9},
        {"Y0": 1e-9, "n": 0.0},
        {"Y0": 1e-1, "n": 1.0},
    ),
    "W": ElementSpec(
        "W", "半无限 Warburg", ("Y0",),
        {"Y0": "S·s^0.5"},
        {"Y0": 1e-6}, {"Y0": 1e-9}, {"Y0": 1e-1},
    ),
}


def parse(text: str) -> tuple:
    """把电路文本解析为节点树，失败时抛 :class:`CircuitError`。"""
    tokens = _tokenize(text)
    pos, node = _parse_series(tokens, 0)
    if pos != len(tokens):
        raise CircuitError(f"第 {pos + 1} 个记号 {tokens[pos]!r} 之后仍有多余内容")
    validate(node)
    return node


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = _TOKEN_RE.match(text, i)
        if not m:
            raise CircuitError(f"位置 {i + 1} 附近无法识别: {text[i:i + 8]!r}")
        tok = m.group(1)
        if tok not in "+|()" and not _ELEMENT_RE.match(tok):
            raise CircuitError(f"未知元件 {tok!r}（仅允许 R/C/L/Q/W 开头）")
        tokens.append(tok)
        i = m.end()
    if not tokens:
        raise CircuitError("空电路")
    return tokens


def _parse_series(tokens: list[str], pos: int) -> tuple[int, tuple]:
    pos, first = _parse_parallel(tokens, pos)
    children = [first]
    while pos < len(tokens) and tokens[pos] == "+":
        pos, child = _parse_parallel(tokens, pos + 1)
        children.append(child)
    node = first if len(children) == 1 else (KIND_SERIES, tuple(children))
    return pos, node


def _parse_parallel(tokens: list[str], pos: int) -> tuple[int, tuple]:
    pos, first = _parse_atom(tokens, pos)
    children = [first]
    while pos < len(tokens) and tokens[pos] == "|":
        pos, child = _parse_atom(tokens, pos + 1)
        children.append(child)
    node = first if len(children) == 1 else (KIND_PARALLEL, tuple(children))
    return pos, node


def _parse_atom(tokens: list[str], pos: int) -> tuple[int, tuple]:
    if pos >= len(tokens):
        raise CircuitError("表达式意外结束")
    tok = tokens[pos]
    if tok == "(":
        pos, node = _parse_series(tokens, pos + 1)
        if pos >= len(tokens) or tokens[pos] != ")":
            raise CircuitError("缺少右括号 ')'")
        return pos + 1, node
    if tok in "+|)":
        raise CircuitError(f"记号 {tok!r} 出现在意外的位置")
    m = _ELEMENT_RE.match(tok)
    return pos + 1, (KIND_ELEMENT, m.group(1).upper(), tok)


def elements(node: tuple) -> list[tuple[str, str]]:
    """按稳定顺序返回 ``(kind, label)`` 列表。"""
    out: list[tuple[str, str]] = []
    _collect(node, out)
    return out


def _collect(node: tuple, out: list[tuple[str, str]]) -> None:
    if node[0] == KIND_ELEMENT:
        out.append((node[1], node[2]))
    else:
        for child in node[1]:
            _collect(child, out)


def validate(node: tuple) -> None:
    labels = [label for _, label in elements(node)]
    if len(set(labels)) != len(labels):
        dup = sorted({x for x in labels if labels.count(x) > 1})
        raise CircuitError(f"元件标签重复: {', '.join(dup)}")
    if not labels:
        raise CircuitError("电路至少要包含一个元件")


_PRECEDENCE = {KIND_SERIES: 0, KIND_PARALLEL: 1, KIND_ELEMENT: 2}


def _emit(node: tuple, parent_prec: int) -> str:
    kind = node[0]
    if kind == KIND_ELEMENT:
        return node[2]
    op = "+" if kind == KIND_SERIES else "|"
    # 子节点只在其优先级低于本层时加括号；展平后同优先级不会作为子节点出现。
    child_threshold = _PRECEDENCE[kind] - 1
    parts = [_emit(c, child_threshold) for c in node[1]]
    text = op.join(parts)
    need_paren = (
        kind in (KIND_SERIES, KIND_PARALLEL)
        and _PRECEDENCE[kind] <= parent_prec
    )
    return f"({text})" if need_paren else text


def render(node: tuple) -> str:
    """节点树 -> 最短、无多余括号的文本（不排序，保留书写顺序）。"""
    return _emit(node, -1)


def _normalize(node: tuple) -> tuple:
    """结合律展平 + 同级子网络按稳定表达字典序排序。"""
    if node[0] == KIND_ELEMENT:
        return (KIND_ELEMENT, node[1], node[2])
    kind = node[0]
    flat: list[tuple] = []
    for child in node[1]:
        norm = _normalize(child)
        if norm[0] == kind:
            flat.extend(norm[1])
        else:
            flat.append(norm)
    kids = sorted(flat, key=_emit_sort_key)
    return (kind, tuple(kids))


def _emit_sort_key(node: tuple) -> tuple:
    # 先按节点种类（元件排在复合网络前），再按文本，保证确定性。
    return (node[0], _emit(node, 0)) if node[0] == KIND_ELEMENT else (node[0], _emit(node, 0))


def canonical(text_or_node) -> str:
    """返回稳定表达：解析、同级排序、最短括号化后的电路文本。"""
    node = text_or_node if isinstance(text_or_node, tuple) else parse(text_or_node)
    return render(_normalize(node))


def default_parameter_table(node: tuple) -> list[dict]:
    """按元件出现顺序给出参数的默认初始值与物理上下界。"""
    table: list[dict] = []
    for kind, label in elements(node):
        spec = ELEMENTS[kind]
        for pname in spec.params:
            table.append({
                "name": f"{label}.{pname}",
                "element": label,
                "param": pname,
                "kind": kind,
                "unit": spec.units[pname],
                "value": spec.defaults[pname],
                "lower": spec.lower[pname],
                "upper": spec.upper[pname],
                "fixed": False,
                "share": "",
            })
    return table
