import math
import numpy as np
import pytest

from pinpai import circuits
from pinpai.circuits import CircuitError


def test_canonical_flatten_and_case():
    assert circuits.parse_canonical("R, C") == "s(r,c)"
    assert circuits.parse_canonical("S(R, P(C, R:ct))") == "s(r,p(c,r:ct))"
    assert circuits.parse_canonical("s(r,s(c,l))") == "s(r,c,l)"
    assert circuits.parse_canonical("p(r,p(c,w))") == "p(r,c,w)"


def test_labeled_and_duplicate():
    assert circuits.parse_canonical("s(r,r:b)") == "s(r,r:b)"
    with pytest.raises(CircuitError):
        circuits.parse("s(r,r)")
    with pytest.raises(CircuitError):
        circuits.parse("s(p(r:ct,cpe:dl),r:ct)")


def test_bad_expressions():
    for bad in ["", "x", "s(", "r,", "p(r", "s(r,,c)", "r:", "c:", "cpe:", "s()", "p(r,)"]:
        with pytest.raises(CircuitError):
            circuits.parse(bad)


def test_passive_sign_convention():
    # 时间因子 e^(+jωt)：电容与 CPE 的 Im(Z)<0；电感 Im>0
    w = np.array([1.0, 10.0, 100.0]) * 2 * math.pi
    zc = circuits.impedance(circuits.parse("c"), {"C": 1e-6}, w)
    assert np.all(zc.real == 0) and np.all(zc.imag < 0)
    zl = circuits.impedance(circuits.parse("l"), {"L": 1e-3}, w)
    assert np.all(zl.imag > 0)
    zq = circuits.impedance(circuits.parse("cpe"), {"Q": 1e-5, "n": 0.9}, w)
    assert np.all(zq.imag < 0)
    # CPE 主值支：n=1 时 Z=1/(jωQ) 即纯电容，相位 -90°
    z1 = circuits.impedance(circuits.parse("cpe"), {"Q": 2e-5, "n": 1.0}, w)
    phase = np.angle(z1)
    assert np.allclose(phase, -math.pi / 2, atol=1e-10)
    zw = circuits.impedance(circuits.parse("w"), {"W": 1.0}, w)
    assert np.allclose(zw.real, -zw.imag, rtol=1e-12)


def test_n_bounds_must_lie_in_unit_interval():
    node = circuits.parse("cpe:dl")
    good = [{"key": "Q:dl", "unit": "S*s^n", "lo": 1e-12, "hi": 1,
             "value": 1e-5, "group": None},
            {"key": "n:dl", "unit": "", "lo": 0.0, "hi": 1.0,
             "value": 0.9, "group": None}]
    assert circuits.validate_param_defs(good, node)
    bad = [{"key": "Q:dl", "unit": "S*s^n", "lo": 1e-12, "hi": 1,
            "value": 1e-5, "group": None},
           {"key": "n:dl", "unit": "", "lo": 0.0, "hi": 1.2,
            "value": 0.9, "group": None}]
    with pytest.raises(CircuitError):
        circuits.validate_param_defs(bad, node)


def test_shared_group_intersection_and_unit():
    node = circuits.parse("s(r,r:b)")
    defs = [
        {"key": "R", "unit": "ohm", "lo": 1, "hi": 100, "value": 10, "group": "g"},
        {"key": "R:b", "unit": "ohm", "lo": 5, "hi": 200, "value": 10, "group": "g"},
    ]
    assert circuits.validate_param_defs(defs, node)
    bad = [dict(d) for d in defs]
    bad[1]["lo"], bad[1]["hi"], bad[1]["value"] = 500, 600, 550
    with pytest.raises(CircuitError):
        circuits.validate_param_defs(bad, node)
