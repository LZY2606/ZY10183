import cmath
import math

import numpy as np
import pytest

from eis.circuit import parse
from eis.impedance import impedance, impedance_simple


def test_capacitor_negative_imaginary():
    z = impedance_simple(parse("C1"), {"C1.C": 1e-6}, 1000.0)
    assert abs(z.real) < 1e-9
    assert z.imag < 0
    assert abs(z.imag + 1.0 / (2 * math.pi * 1000 * 1e-6)) < 1e-12


def test_inductor_positive_imaginary():
    z = impedance_simple(parse("L1"), {"L1.L": 1e-3}, 1000.0)
    assert z.imag > 0


def test_cpe_principal_branch_and_n1_is_capacitor():
    node = parse("Q1")
    z_n1 = impedance_simple(node, {"Q1.Y0": 1e-5, "Q1.n": 1.0}, 1234.0)
    z_cap = impedance_simple(parse("C1"), {"C1.C": 1e-5}, 1234.0)
    assert abs(z_n1 - z_cap) < 1e-12
    # n=0.5 与 Warburg 同形（主值支）
    z_half = impedance_simple(node, {"Q1.Y0": 2e-4, "Q1.n": 0.5}, 50.0)
    w = 1.0 / (2e-4 * cmath.exp(0.5 * (cmath.log(2 * math.pi * 50)
                                      + 1j * math.pi / 2)))
    assert abs(z_half - w) < 1e-12
    # 0<n<1 虚部为负
    assert impedance_simple(node, {"Q1.Y0": 1e-6, "Q1.n": 0.8}, 80.0).imag < 0


def test_warburg_formula():
    # Z_W = (1-j)/(Y0 sqrt(2 omega))，Re=-Im=1/(Y0 sqrt(2ω))
    z = impedance_simple(parse("W1"), {"W1.Y0": 1e-3}, 31.83)
    mag = 1.0 / (1e-3 * math.sqrt(2 * 2 * math.pi * 31.83))
    assert abs(z.real - mag) < 1e-4
    assert abs(z.imag + mag) < 1e-4


def test_series_and_parallel_rules():
    z = impedance_simple(parse("R1+C1"), {"R1.R": 100, "C1.C": 1e-6}, 1591.5)
    assert z.real == pytest.approx(100.0)
    assert z.imag < 0
    zp = impedance_simple(parse("R1|C1"), {"R1.R": 100, "C1.C": 1e-6}, 1591.5)
    # f=1/(2πRC) 时 Zc=-jR，并联 -> R(1-j)/2
    assert zp.real == pytest.approx(50.0, rel=1e-3)
    assert zp.imag == pytest.approx(-50.0, rel=1e-3)


def test_nonpositive_frequency_rejected():
    with pytest.raises(ValueError):
        impedance(parse("R1"), {"R1.R": 1.0}, np.array([0.0, 1.0]))
