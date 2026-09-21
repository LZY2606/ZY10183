import pytest

from eis.circuit import (
    CircuitError, canonical, default_parameter_table, elements, parse, render,
)


def test_associative_canonical_equal():
    pairs = [
        ("Rs+(Rct|Cdl)", "(Rct|Cdl)+Rs"),
        ("R1+R2+R3", "(R1+R2)+R3"),
        ("(C1|R1)|Q1", "Q1|(R1|C1)"),
        ("(R1+Rs)|C1", "C1|(Rs+R1)"),
    ]
    canon = [canonical(a) for a, _ in pairs]
    for (a, b), ca in zip(pairs, canon):
        assert ca == canonical(b), (a, b, ca)


def test_different_topology_distinct():
    assert canonical("R1|C1") != canonical("R1+C1")
    assert canonical("Rs+(Rct|Cdl)") != canonical("Rs+Rct+(Rleak|Cdl)")


def test_minimal_parentheses_roundtrip():
    # | 优先于 +，括号在此冗余；最短表达去掉括号但语义不变。
    assert render(parse("Rs+(Rct|Cdl)")) == "Rs+Rct|Cdl"
    assert canonical("Rs+(Rct|Cdl)") == canonical("Rs+Rct|Cdl")
    assert "(" not in canonical("R1+R2+R3")


def test_elements_and_defaults():
    node = parse("Rs+(Rct|Q1)")
    labs = [l for _, l in elements(node)]
    assert labs == ["Rs", "Rct", "Q1"]
    table = default_parameter_table(node)
    names = [p["name"] for p in table]
    assert names == ["Rs.R", "Rct.R", "Q1.Y0", "Q1.n"]
    qn = next(p for p in table if p["name"] == "Q1.n")
    assert qn["lower"] == 0.0 and qn["upper"] == 1.0


def test_reject_bad_text():
    with pytest.raises(CircuitError):
        parse("R1 + ")
    with pytest.raises(CircuitError):
        parse("(R1+C1")
    with pytest.raises(CircuitError):
        parse("R1|R1")  # 重复标签


def test_warburg_and_inductor_parse():
    node = parse("L1+W1")
    assert {l for _, l in elements(node)} == {"L1", "W1"}
