"""验收：观测带内两个近等价电路、一个近奇异组合、贴界≠收敛、权重子版本。"""
import numpy as np

from pinpai import fixtures, fitting


def _fit(expr, defs, **kw):
    freqs, z = fixtures.synthetic_dataset()
    kw.setdefault("weight_mode", "modulus")
    kw.setdefault("n_starts", 12)
    kw.setdefault("max_iter", 300)
    return fitting.run_fit(expr, defs, freqs, z, **kw)


def test_two_circuits_near_equivalent_in_band():
    a = _fit(fixtures.CANDIDATE_A_EXPR, fixtures.CANDIDATE_A_PARAMS)
    b = _fit(fixtures.CANDIDATE_B_EXPR, fixtures.CANDIDATE_B_PARAMS)
    # 两模型的加权 SSE 处于同一量级（观测频带内 Nyquist 弧几乎相同）
    assert a["sse"] < 0.5
    assert b["sse"] < 0.5
    assert abs(a["sse"] - b["sse"]) / max(a["sse"], b["sse"]) < 0.2
    # 但结构与参数不同（A 是理想电容，B 是 CPE）
    assert "C" in a["values"]
    assert "Q:dl" in b["values"]


def test_candidate_A_converges_interior_with_se():
    a = _fit(fixtures.CANDIDATE_A_EXPR, fixtures.CANDIDATE_A_PARAMS)
    assert a["reason"] == "converged"
    assert not any(s["at_upper"] or s["at_lower"]
                   for s in a["bound_states"])
    assert not a["diagnosis"]["near_rank_deficient"]
    assert a["diagnosis"]["se"] is not None
    assert set(a["diagnosis"]["se"]) == set(a["values"])


def test_candidate_B_n_presses_upper_bound_not_converged():
    b = _fit(fixtures.CANDIDATE_B_EXPR, fixtures.CANDIDATE_B_PARAMS)
    # 终止原因必须是贴界，且明确指出是 n:dl 贴上界
    assert b["reason"] == "at_bound"
    n_state = next(s for s in b["bound_states"] if s["key"] == "n:dl")
    assert n_state["at_upper"] and abs(n_state["value"] - n_state["hi"]) < 1e-9
    # 多起点并排结果中，最佳解（rank 0）带贴上界标记
    best = next(s for s in b["starts"] if s["rank"] == 0)
    assert "n:dl" in best["at_upper"]
    assert best["reason"] == "at_bound"
    # 至少有一个起点报告真正收敛（在不同局部极小），界面才能并排比较
    assert any(s["reason"] == "converged" for s in b["starts"])


def test_candidate_C_near_singular_no_fake_standard_errors():
    c = _fit(fixtures.CANDIDATE_C_EXPR, fixtures.CANDIDATE_C_PARAMS)
    d = c["diagnosis"]
    assert d["near_rank_deficient"]
    assert d["se"] is None            # 拒绝伪精密标准误
    assert d["correlation"] is None
    assert d["null_directions"]
    # 弱方向同时包含 R 与 R:b（二者之和才可辨识），分量符号相反
    nd = d["null_directions"][0]
    assert "R" in nd and "R:b" in nd
    assert nd["R"] * nd["R:b"] < 0
    assert abs(abs(nd["R"]) - abs(nd["R:b"])) < 0.02
    # C 自身拟合很好（弧几乎相同），但参数分配不可信
    assert c["sse"] < 0.05


def test_fixed_param_rerun_locks_value():
    b = _fit(fixtures.CANDIDATE_B_EXPR, fixtures.CANDIDATE_B_PARAMS)
    # 模拟 API refit：把 n 固定在 0.6（内部点），界收缩后所有起点都该锁死
    import copy
    defs = copy.deepcopy(fixtures.CANDIDATE_B_PARAMS)
    for d in defs:
        if d["key"] == "n:dl":
            d["lo"], d["hi"], d["value"] = 0.6 * (1 - 1e-8), 0.6 * (1 + 1e-8), 0.6
    r = _fit(fixtures.CANDIDATE_B_EXPR, defs, n_starts=5)
    assert abs(r["values"]["n:dl"] - 0.6) < 1e-7
    state = next(s for s in r["bound_states"] if s["key"] == "n:dl")
    assert state["at_upper"] or state["at_lower"]


def test_frequency_weight_changes_child_version_traceably():
    g0 = _fit(fixtures.CANDIDATE_A_EXPR, fixtures.CANDIDATE_A_PARAMS, gamma=0.0)
    g1 = _fit(fixtures.CANDIDATE_A_EXPR, fixtures.CANDIDATE_A_PARAMS, gamma=0.5)
    assert g0["weight"]["gamma"] == 0.0
    assert g1["weight"]["gamma"] == 0.5
    # 逐点权重确实不同（高频权重被提升）
    w0 = np.array([p["weight"] for p in g0["points"]])
    w1 = np.array([p["weight"] for p in g1["points"]])
    ratio = w1 / w0
    f = np.array([p["freq_hz"] for p in g0["points"]])
    assert ratio[np.argmax(f)] > ratio[np.argmin(f)]


def test_multistart_results_kept_side_by_side():
    a = _fit(fixtures.CANDIDATE_A_EXPR, fixtures.CANDIDATE_A_PARAMS, n_starts=8)
    assert len(a["starts"]) == 8
    ranks = sorted(s["rank"] for s in a["starts"])
    assert ranks == list(range(8))
    sses = sorted(s["sse"] for s in a["starts"])
    assert sses[0] == a["sse"]
