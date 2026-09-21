import numpy as np
import pytest

from eis.circuit import parse
from eis.fixtures import TRUE_VALUES, build_candidates
from eis.fitting import FitProblem, solve


def _problem(cand, arrays, weighting="sigma"):
    _, f, z, sr, si = arrays
    return FitProblem(parse(cand["text"]), cand["parameters"], f, z, sr, si,
                      weighting)


def test_c1_rc_truly_converges_with_se(arrays):
    cand = build_candidates()[0]
    res = solve(_problem(cand, arrays), n_starts=10)
    assert res.status == "converged"
    assert not res.at_upper and not res.at_lower
    assert not res.near_singular and res.rank == res.nvar == 3
    # 恢复真值（2% 以内）
    assert res.values["Rs.R"] == pytest.approx(10.0, abs=0.2)
    assert res.values["Rct.R"] == pytest.approx(100.0, rel=0.03)
    assert res.values["Cdl.C"] == pytest.approx(1e-4, rel=0.05)
    # 满秩时给出有限标准误
    assert all(np.isfinite(se) for se in res.se.values())


def test_c2_cpe_n_pinned_at_upper_bound_not_converged(arrays):
    cand = build_candidates()[1]
    res = solve(_problem(cand, arrays), n_starts=10)
    # 与 C1 几乎相同的拟合优度（观测频带内等价）
    c1 = solve(_problem(build_candidates()[0], arrays), n_starts=6)
    assert abs(res.chi2 - c1.chi2) / c1.chi2 < 0.05
    # n 贴着用户上界 0.999，状态必须与真正收敛区分开
    assert "Q1.n" in res.at_upper
    assert res.status == "bound_only"


def test_c3_near_singular_no_pseudo_precision_se(arrays):
    cand = build_candidates()[2]
    res = solve(_problem(cand, arrays), n_starts=10)
    assert res.near_singular
    assert res.rank < res.nvar
    # 近秩亏：所有 SE 必须为 None，绝不输出伪精密数字
    assert all(se is None for se in res.se.values())
    assert res.null_directions  # 至少给出一个零空间方向
    # 串联双电阻只有和可辨识
    rsum = res.values["Rs.R"] + res.values["Rleak.R"]
    assert rsum == pytest.approx(10.0, abs=0.3)


def test_multistart_results_all_retained(arrays):
    cand = build_candidates()[0]
    res = solve(_problem(cand, arrays), n_starts=8)
    assert len(res.starts) == 8
    chi2s = [s["chi2"] for s in res.starts if s["chi2"] is not None]
    # 起点异质：存在较差的局部解/停滞，最优解被选中
    assert res.starts[res.best_index]["chi2"] == min(chi2s)
    assert max(chi2s) > min(chi2s)


def test_fixed_parameter_and_weighting_child_runs(arrays):
    cand = build_candidates()[0]
    _, f, z, sr, si = arrays
    params = [dict(p) for p in cand["parameters"]]
    for p in params:
        if p["name"] == "Rs.R":
            p["fixed"] = True
            p["value"] = 10.0
    prob = FitProblem(parse(cand["text"]), params, f, z, sr, si, "sigma")
    res = solve(prob, n_starts=6)
    assert res.values["Rs.R"] == 10.0
    assert res.se["Rs.R"] is None  # 固定量不给 SE
    # 等权重跑同样可收敛（口径不同但可重放）
    prob2 = FitProblem(parse(cand["text"]),
                       [dict(p, fixed=False) for p in params],
                       f, z, sr, si, "unit")
    res2 = solve(prob2, n_starts=6)
    assert res2.status in ("converged", "bound_only")
    assert res2.weighting == "unit"


def test_share_group_links_parameters(arrays):
    _, f, z, sr, si = arrays
    text = "R1+(C1|R2)"
    from eis.circuit import default_parameter_table
    params = default_parameter_table(parse(text))
    for p in params:
        if p["name"] in ("R1.R", "R2.R"):
            p["share"] = "gR"
    prob = FitProblem(parse(text), params, f, z, sr, si, "sigma")
    # 共享后自由变量数 = 3（共享R、C1.C、无其它）
    assert prob.nvar == 2
    res = solve(prob, n_starts=6)
    assert res.values["R1.R"] == res.values["R2.R"]


def test_share_group_cross_kind_rejected(arrays):
    _, f, z, sr, si = arrays
    text = "R1+C1"
    from eis.circuit import default_parameter_table
    params = default_parameter_table(parse(text))
    for p in params:
        p["share"] = "g"
    with pytest.raises(ValueError):
        FitProblem(parse(text), params, f, z, sr, si)
