"""加权复阻抗拟合：有界 Levenberg–Marquardt、多起点、协方差秩亏诊断。

只依赖 NumPy。设计要点（与 README“数据口径”一致）：

* 残差向量 ``r = sqrt(w) * [Re(Z_m-Z_d); Im(Z_m-Z_d)]``（先实后虚）。
* 权重模式：
  - ``unit``     ：w = 1；
  - ``modulus``  ：w = 1/|Z|^2（EIS 常用比例权重）；
  - ``sigma``    ：w = 1/sigma^2，逐点给定且实虚部共用。
  - 额外乘子 f_gamma = (f/f_ref)^gamma，默认 gamma=0；非零 gamma 即“频率权重”。
* 正物理量（R/C/L/Q/W）在 **对数空间** 优化，CPE 指数 n 在线性空间优化；
  共享组只保留一个自由变量。
* 有界处理采用内部变量 + 双端 sigmoid 映射，梯度近零时若变量仍在界上，
  终止原因记为 ``at_bound``（贴界）而非 ``converged``（真正收敛）。
* 设计矩阵（数值中心差 Jacobian）经 SVD 判秩：有效秩不足时不给标准误，
  只给相关方向/零空间方向与条件数，避免伪精密。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import circuits

LOG_BASES = {"R", "C", "L", "Q", "W"}
MAX_LAMBDA = 1e12
BOUND_TOL_FACTOR = 1e-6  # 距界相对距离小于此值视为贴界


@dataclass
class Var:
    idx: int                 # 自由变量序号
    keys: List[str]          # 该变量绑定的物理参数（共享组可能多个）
    log: bool
    lo: np.ndarray
    hi: np.ndarray

    def to_int(self, phys: Dict[str, float]) -> np.ndarray:
        vals = np.array([phys[k] for k in self.keys], dtype=float)
        if self.log:
            return np.log(vals)
        return vals

    def to_phys_single(self, x: float) -> float:
        if self.log:
            return math.exp(x)
        return x


def _intersect(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> Tuple[float, float]:
    lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
    if not lo < hi:
        raise circuits.CircuitError("共享组成员的参数界交集为空")
    return lo, hi


def build_variables(param_defs: List[dict]) -> List[Var]:
    """根据共享组把物理参数压缩为自由变量；共享参数取交集界。"""
    by_group: Dict[str, List[dict]] = {}
    singles: List[dict] = []
    for pd in param_defs:
        if pd.get("group"):
            by_group.setdefault(pd["group"], []).append(pd)
        else:
            singles.append(pd)
    groups = [
        {"group": g, "members": members}
        for g, members in sorted(by_group.items())
    ]
    ordered = singles + groups
    ordered.sort(key=lambda d: min(param_defs.index(p) for p in (
        d["members"] if "members" in d else [d])))

    variables: List[Var] = []
    for i, item in enumerate(ordered):
        if "members" in item:
            members = item["members"]
            keys = [m["key"] for m in members]
            base = keys[0].split(":", 1)[0]
            lo = members[0]["lo"]
            hi = members[0]["hi"]
            for m in members[1:]:
                lo, hi = _intersect(lo, hi, m["lo"], m["hi"])
            value = members[0]["value"]
            for m in members[1:]:
                if abs(m["value"] - value) > 1e-12 * max(1.0, abs(value)):
                    raise circuits.CircuitError(
                        f"共享组 {item['group']} 的初值必须一致（{keys}）")
            log = base in LOG_BASES
            lo_arr = np.full(len(keys), (math.log(lo) if log else lo))
            hi_arr = np.full(len(keys), (math.log(hi) if log else hi))
        else:
            pd = item
            keys = [pd["key"]]
            base = pd["key"].split(":", 1)[0]
            log = base in LOG_BASES
            lo, hi = pd["lo"], pd["hi"]
            lo_arr = np.array([math.log(lo) if log else lo])
            hi_arr = np.array([math.log(hi) if log else hi])
        variables.append(Var(idx=i, keys=keys, log=log, lo=lo_arr, hi=hi_arr))
    return variables


# --------------------------------------------------------------------------- #
# 变量映射：内部无界 u <-> 有界 x
# --------------------------------------------------------------------------- #
def _u_to_x(u: float, var: Var) -> np.ndarray:
    """标量 u -> 该变量绑定的每个物理参数的有界内部值（盒域投影裁剪）。"""
    lo, hi = float(var.lo[0]), float(var.hi[0])
    x = min(max(float(u), lo), hi)
    return np.full(var.lo.shape, x, dtype=float)


class _Mapper:
    def __init__(self, variables: Sequence[Var]):
        self.variables = list(variables)
        self.nv = len(variables)

    def initial_u(self, param_defs: List[dict]) -> np.ndarray:
        phys = {pd["key"]: float(pd["value"]) for pd in param_defs}
        u = np.zeros(self.nv)
        for var in self.variables:
            x = var.to_int(phys)
            u[var.idx] = float(np.mean(x))
        return u

    def to_values(self, u: np.ndarray) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for var in self.variables:
            x = _u_to_x(u[var.idx], var)
            for j, key in enumerate(var.keys):
                values[key] = math.exp(x[j]) if var.log else x[j]
        return values

    def bound_state(self, u: np.ndarray) -> List[dict]:
        states = []
        for var in self.variables:
            x = _u_to_x(u[var.idx], var)
            for j, key in enumerate(var.keys):
                lo, hi = var.lo[j], var.hi[j]
                width = max(hi - lo, 1e-30)
                at_lo = (x[j] - lo) <= BOUND_TOL_FACTOR * max(1.0, abs(lo), width)
                at_hi = (hi - x[j]) <= BOUND_TOL_FACTOR * max(1.0, abs(hi), width)
                states.append({
                    "key": key, "at_lower": bool(at_lo), "at_upper": bool(at_hi),
                    "lo": math.exp(lo) if var.log else lo,
                    "hi": math.exp(hi) if var.log else hi,
                    "value": math.exp(x[j]) if var.log else x[j],
                })
        return states


# --------------------------------------------------------------------------- #
# 权重与残差
# --------------------------------------------------------------------------- #
def point_weights(freqs: np.ndarray, z_data: np.ndarray,
                  weight_mode: str, sigma: Optional[np.ndarray],
                  gamma: float, f_ref: Optional[float]) -> np.ndarray:
    if weight_mode == "unit":
        w = np.ones_like(freqs)
    elif weight_mode == "modulus":
        w = 1.0 / np.abs(z_data) ** 2
    elif weight_mode == "sigma":
        if sigma is None:
            raise ValueError("sigma 权重模式需要提供 sigma")
        s = np.asarray(sigma, dtype=float)
        if np.any(~np.isfinite(s)) or np.any(s <= 0):
            raise ValueError("sigma 必须为正有限值")
        w = 1.0 / s ** 2
    else:
        raise ValueError(f"未知权重模式 {weight_mode}")
    if gamma:
        ref = f_ref if f_ref is not None else float(np.exp(np.mean(np.log(freqs))))
        w = w * (freqs / ref) ** (2.0 * gamma)
    return w


def residual(values: Dict[str, float], node, freqs: np.ndarray,
             z_data: np.ndarray, sqrt_w: np.ndarray) -> np.ndarray:
    z_model = circuits.impedance(node, values, 2.0 * math.pi * freqs)
    dz = z_model - z_data
    return np.concatenate([sqrt_w * dz.real, sqrt_w * dz.imag])


def _jacobian(u: np.ndarray, mapper: _Mapper, node, freqs, z_data, sqrt_w,
              step: float = 1e-6) -> np.ndarray:
    n = 2 * freqs.size
    jac = np.zeros((n, mapper.nv))
    for k in range(mapper.nv):
        h = step * max(1.0, abs(u[k]))
        up = u.copy(); up[k] += h
        dn = u.copy(); dn[k] -= h
        rp = residual(mapper.to_values(up), node, freqs, z_data, sqrt_w)
        rn = residual(mapper.to_values(dn), node, freqs, z_data, sqrt_w)
        jac[:, k] = (rp - rn) / (2.0 * h)
    return jac


def sse_of(r: np.ndarray) -> float:
    return float(r @ r)


# --------------------------------------------------------------------------- #
# 有界 LM：硬投影（裁剪）+ 活动集 KKT 判定
# --------------------------------------------------------------------------- #
def _u_to_x_vec(u: np.ndarray, mapper: _Mapper) -> np.ndarray:
    lo, hi = _lo_vec(mapper), _hi_vec(mapper)
    return np.minimum(np.maximum(u, lo), hi)


def _u_to_x(u, var: Var):
    """标量 u -> 该变量绑定的每个物理参数的有界内部值（投影裁剪）。"""
    lo, hi = float(var.lo[0]), float(var.hi[0])
    x = min(max(float(u), lo), hi)
    return np.full(var.lo.shape, x, dtype=float)


def lm_solve(u0: np.ndarray, mapper: _Mapper, node, freqs, z_data, sqrt_w,
             max_iter: int = 200, tol: float = 1e-9) -> dict:
    """投影 LM。

    * 步被裁剪到盒域内；
    * ``converged``：无活动方向上梯度足够小（真正内部 KKT 收敛）；
    * ``at_bound``：无法再下降，且某自由变量贴界且其活动方向梯度仍显著
      （即“被上界挡住”，而不是数值迭代收敛）；
    * ``max_iter``：达到迭代上限仍在尝试移动。
    """
    lo, hi = _lo_vec(mapper), _hi_vec(mapper)
    width = np.maximum(hi - lo, 1e-30)
    u = np.clip(np.array(u0, dtype=float), lo + 1e-9 * width, hi - 1e-9 * width)
    r = residual(mapper.to_values(u), node, freqs, z_data, sqrt_w)
    sse = sse_of(r)
    lam = 1e-3
    reason = "max_iter"
    it = 0
    for it in range(1, max_iter + 1):
        jac = _jacobian(u, mapper, node, freqs, z_data, sqrt_w)
        grad = jac.T @ r
        near_lo = (u - lo) <= BOUND_TOL_FACTOR * width
        near_hi = (hi - u) <= BOUND_TOL_FACTOR * width
        # 活动方向：贴下界且梯度要求继续减小 x（grad>0 想往下走），反之亦然
        blocked = (near_lo & (grad > 0)) | (near_hi & (grad < 0))
        free = ~blocked
        near_any = near_lo | near_hi
        g_free = grad[free]
        gnorm = float(np.linalg.norm(g_free, ord=np.inf)) if g_free.size else 0.0
        if gnorm < 1e-7 * max(1.0, sse):
            reason = "at_bound" if np.any(near_any) else "converged"
            break
        moved = False
        u_prev = u.copy()
        sse_prev = sse
        jtj = jac.T @ jac
        for _ in range(80):
            a = jtj + lam * np.diag(np.diag(jtj) + 1e-12)
            try:
                du = -np.linalg.solve(a, grad)
            except np.linalg.LinAlgError:
                du = -np.linalg.lstsq(a, grad, rcond=None)[0]
            du = np.where(free, du, 0.0)
            u_new = np.clip(u + du, lo, hi)
            if float(np.linalg.norm(u_new - u)) == 0.0:
                break
            r_new = residual(mapper.to_values(u_new), node, freqs, z_data, sqrt_w)
            sse_new = sse_of(r_new)
            if np.isfinite(sse_new) and sse_new < sse * (1 - 1e-12):
                u, r, sse = u_new, r_new, sse_new
                lam = max(lam * 0.3, 1e-12)
                moved = True
                break
            lam *= 2.0
            if lam > MAX_LAMBDA:
                break
        if not moved:
            blocked_grad = np.abs(grad[blocked])
            bound_push = (blocked_grad.size and
                          float(np.max(blocked_grad)) > 1e-3 * max(1.0, sse))
            if bound_push and np.any(near_any):
                reason = "at_bound"
            elif gnorm < 1e-6 * max(1.0, sse):
                reason = "converged"
            else:
                reason = "max_iter"
            break
        # 步几乎全被界吃掉且 SSE 基本不再下降：贴界停滞
        actual = float(np.linalg.norm((u - u_prev) / np.maximum(width, 1e-30),
                                      ord=np.inf))
        rel_gain = (sse_prev - sse) / max(1.0, sse_prev)
        blocked_grad = np.abs(grad[blocked])
        bound_push = (blocked_grad.size and
                      float(np.max(blocked_grad)) > 1e-3 * max(1.0, sse))
        # SSE 增益可忽略，且自由方向梯度已很小。若界方向仍有显著“想越界”
        # 的梯度（比自由梯度大一两个量级），说明是被界挡住（贴界），
        # 而非数值迭代真正收敛。
        bg = float(np.max(np.abs(grad[blocked]))) if np.any(blocked) else 0.0
        if rel_gain < 1e-6 and gnorm < 1e-2 * max(1.0, bg, sse * 1e-3):
            if bg > 10.0 * max(gnorm, 1e-12) and np.any(near_any):
                reason = "at_bound"
            else:
                reason = "converged"
            break
    return {"u": u, "sse": sse, "iterations": it, "reason": reason}


def _lo_vec(mapper: _Mapper) -> np.ndarray:
    return np.array([float(v.lo[0]) for v in mapper.variables])


def _hi_vec(mapper: _Mapper) -> np.ndarray:
    return np.array([float(v.hi[0]) for v in mapper.variables])


# --------------------------------------------------------------------------- #
# 协方差 / 可辨识性诊断
# --------------------------------------------------------------------------- #
def covariance_diagnosis(u: np.ndarray, mapper: _Mapper, node, freqs,
                         z_data, sqrt_w, r: np.ndarray) -> dict:
    """在最优解处做 Gauss–Newton 协方差与设计矩阵秩诊断。

    Jacobian 先按内部变量尺度归一（对数变量即相对灵敏度），再做 SVD。
    有效秩 < 自由参数数时返回 ``near_rank_deficient`` 与零空间方向，
    不输出标准误。
    """
    jac = _jacobian(u, mapper, node, freqs, z_data, sqrt_w)
    n_res, n_par = jac.shape
    dof = max(n_res - n_par, 1)
    s2 = sse_of(r) / dof
    x = np.array([float(_u_to_x(u[v.idx], v)[0]) for v in mapper.variables])
    # 物理量直接灵敏度：内部为对数变量时 d r/dln p -> d r/dp 需除以 p；
    # 线性变量（n）保持不变。这样零空间向量直接表示物理参数的变化方向。
    phys = np.array([
        math.exp(x[v.idx]) if v.log else x[v.idx] for v in mapper.variables])
    col_scale = np.where([v.log for v in mapper.variables],
                         1.0 / np.maximum(phys, 1e-300), 1.0)
    js = jac * col_scale
    _u_mat, s, vt = np.linalg.svd(js, full_matrices=False)
    cutoff = max(n_res, n_par) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    good = s > cutoff
    rank = int(np.sum(good))
    full_rank = rank == n_par
    cond_scaled = float(s[0] / s[-1]) if s.size and s[-1] > cutoff else math.inf
    # 机器精度秩之外再设“可辨识性阈值”：条件数超过 1e8 时，形式上满秩但
    # 标准误会被噪声放大到不可信，按近秩亏处理，拒绝给出伪精密 SE。
    PRACTICAL_COND = 1e8
    practically_singular = (not full_rank) or (
        math.isfinite(cond_scaled) and cond_scaled > PRACTICAL_COND)
    weak = s <= (s[0] / PRACTICAL_COND)
    result = {
        "dof": dof,
        "rank": rank,
        "n_free": n_par,
        "condition": cond_scaled if math.isfinite(cond_scaled) else None,
        "singular_values": [float(v) for v in s],
        "se": None,
        "correlation": None,
        "near_rank_deficient": bool(practically_singular),
        "null_directions": None,
    }
    keys: List[str] = []
    for var in mapper.variables:
        keys.extend(var.keys)
    if not practically_singular:
        # 显式：cov = V S^-2 V^T
        cov_s = (vt.T * (1.0 / s ** 2)) @ vt
        cov_s *= s2
        d = np.sqrt(np.diag(cov_s))
        corr = cov_s / np.maximum(np.outer(d, d), 1e-300)
        se_map, cm = {}, np.ones((len(keys), len(keys)))
        for var in mapper.variables:
            se_value = float(d[var.idx])
            for key in var.keys:
                se_map[key] = se_value
        for a, va in enumerate(mapper.variables):
            for b, vb in enumerate(mapper.variables):
                c = float(np.clip(corr[a, b], -1.0, 1.0))
                for ka in va.keys:
                    for kb in vb.keys:
                        cm[keys.index(ka), keys.index(kb)] = c
        result["se"] = se_map
        result["correlation"] = {"keys": keys, "matrix": cm.tolist()}
    else:
        # 机器零空间或实际不可辨识的最弱方向都报出，供人工辨证
        rows = vt[~good] if not full_rank else vt[weak]
        # vt 是“尺度归一 Jacobian”的奇异向量：对数变量的分量对应
        # 物理量的相对变化方向（dln p），线性变量对应绝对变化。
        dirs = []
        for row in rows:
            norm = float(np.linalg.norm(row)) or 1.0
            entry = {}
            for k in range(n_par):
                comp = float(row[k] / norm)
                if abs(comp) > 1e-3:
                    entry[mapper.variables[k].keys[0]] = comp
            dirs.append(entry)
        result["null_directions"] = dirs
    return result


# --------------------------------------------------------------------------- #
# 多起点拟合总控
# --------------------------------------------------------------------------- #
def _multistart_seeds(mapper: _Mapper, n_starts: int,
                      rng: np.random.Generator) -> List[np.ndarray]:
    lo, hi = _lo_vec(mapper), _hi_vec(mapper)
    margin = 0.02 * (hi - lo)
    seeds: List[np.ndarray] = []
    for _ in range(n_starts):
        u = np.empty(mapper.nv)
        for k, var in enumerate(mapper.variables):
            u[k] = rng.uniform(lo[k] + margin[k], hi[k] - margin[k])
        seeds.append(u)
    return seeds


def run_fit(expr: str, param_defs: List[dict], freqs: np.ndarray,
            z_data: np.ndarray, *, weight_mode: str = "modulus",
            sigma: Optional[np.ndarray] = None, gamma: float = 0.0,
            f_ref: Optional[float] = None, n_starts: int = 8, seed: int = 1186,
            max_iter: int = 200) -> dict:
    node = circuits.parse(expr)
    param_defs = circuits.validate_param_defs(param_defs, node)
    variables = build_variables(param_defs)
    mapper = _Mapper(variables)
    w = point_weights(freqs, z_data, weight_mode, sigma, gamma, f_ref)
    sqrt_w = np.sqrt(w)
    rng = np.random.default_rng(seed)
    seeds = _multistart_seeds(mapper, n_starts, rng)

    starts = []
    best = None
    best_seed = 0
    for si, u0 in enumerate(seeds):
        sol = lm_solve(u0, mapper, node, freqs, z_data, sqrt_w,
                       max_iter=max_iter)
        values = mapper.to_values(sol["u"])
        states = {s["key"]: s for s in mapper.bound_state(sol["u"])}
        starts.append({
            "start": si,
            "sse": sol["sse"],
            "iterations": sol["iterations"],
            "reason": sol["reason"],
            "values": {k: float(v) for k, v in values.items()},
            "at_upper": sorted(k for k, s in states.items() if s["at_upper"]),
            "at_lower": sorted(k for k, s in states.items() if s["at_lower"]),
        })
        if best is None or sol["sse"] < best["sse"]:
            best, best_seed = sol, si

    starts.sort(key=lambda s: s["sse"])
    for rank_i, s in enumerate(starts):
        s["rank"] = rank_i
    u = best["u"]
    values = mapper.to_values(u)
    z_model = circuits.impedance(node, values, 2.0 * math.pi * freqs)
    r = residual(values, node, freqs, z_data, sqrt_w)
    diag = covariance_diagnosis(u, mapper, node, freqs, z_data, sqrt_w, r)
    states = mapper.bound_state(u)

    free_groups = [{"keys": var.keys, "log": var.log} for var in variables]
    points = []
    for i, f in enumerate(freqs):
        points.append({
            "freq_hz": float(f),
            "zreal_data": float(z_data.real[i]),
            "zimag_data": float(z_data.imag[i]),
            "zreal_fit": float(z_model.real[i]),
            "zimag_fit": float(z_model.imag[i]),
            "resid_real": float(sqrt_w[i] * (z_model.real[i] - z_data.real[i])),
            "resid_imag": float(sqrt_w[i] * (z_model.imag[i] - z_data.imag[i])),
            "weight": float(w[i]),
        })
    return {
        "expr_canonical": circuits.canonical(node),
        "param_defs": param_defs,
        "weight": {"mode": weight_mode, "gamma": float(gamma),
                   "f_ref": float(f_ref) if f_ref else
                   float(np.exp(np.mean(np.log(freqs))))},
        "sse": best["sse"],
        "reason": best["reason"],
        "iterations": best["iterations"],
        "best_start": best_seed,
        "values": {k: float(v) for k, v in values.items()},
        "bound_states": states,
        "free_groups": free_groups,
        "diagnosis": diag,
        "starts": starts,
        "points": points,
    }
