"""复阻抗加权非线性最小二乘拟合（纯 NumPy）。

目标（每点残差按测量标准差归一）::

    r_{i,Re} = (Re Z_model(f_i) - z_re_i) / sigma_re_i
    r_{i,Im} = (Im Z_model(f_i) - z_im_i) / sigma_im_i
    chi2 = sum(r**2)

支持三种权重（``weighting``）：

* ``sigma``  —— 用数据列 sigma_re/sigma_im；缺列时退化为等权；
* ``unit``   —— sigma=1（原始欧姆量纲的最小二乘）；
* ``modulus``—— sigma_i = |Z_obs_i|（幅模加权，降低高阻抗点的支配性）。

有界参数经 logit 映射到无界优化变量 t：

    x = lo + (hi - lo) * sigmoid(t)
    t = logit((x - lo)/(hi - lo))

因此 CPE 指数 n 的用户边界（默认 [0,1]）与 R/C 参数边界统一处理，
边界约束在任何一次残差计算中都不会被违反。

收敛与“贴边界”是两回事，本模块分别报告：

* converged: 梯度与步长阈值同时满足，且没有自由参数贴在边界；
* bound:     有自由参数位于边界邻域（rel < 1e-6 或 > 1-1e-6），
             此时即便残差梯度很小，也只标记为 ``bound_only``，
             提醒“优化想继续但被边界挡住”，不宣称自由收敛；
* max_iter:  迭代次数用尽。

协方差 / 标准误用设计矩阵 J=dr/dx（物理量纲，列已按参数区间标定）的
SVD 计算：数值秩缺（最小奇异值 < 1e-8 * 最大奇异值，约对应近奇异
方向上灵敏度低于主方向 1 亿分之一）时**不输出任何标准误**，
只给出相关方向与零空间信息，避免伪精密。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .circuit import elements
from .impedance import impedance

SIGMOID_LIMIT = 36.0
BOUND_FRAC = 1e-6
SVD_RANK_CUTOFF = 1e-8
MAX_ITERS = 200


@dataclass
class FitProblem:
    node: tuple
    parameters: list[dict]
    freqs: np.ndarray
    z_obs: np.ndarray  # complex
    sigma_re: np.ndarray
    sigma_im: np.ndarray
    weighting: str = "unit"

    def __post_init__(self):
        self.freqs = np.asarray(self.freqs, dtype=float)
        self.z_obs = np.asarray(self.z_obs, dtype=np.complex128)
        self.sigma_re = np.asarray(self.sigma_re, dtype=float)
        self.sigma_im = np.asarray(self.sigma_im, dtype=float)
        self._build_variables()

    def _build_variables(self):
        labels = {label for _, label in elements(self.node)}
        for p in self.parameters:
            if p["element"] not in labels:
                raise ValueError(f"参数 {p['name']} 不属于电路中的元件")
        groups: dict[str, list[dict]] = {}
        free_rows: list[dict] = []
        for p in self.parameters:
            share = (p.get("share") or "").strip()
            if share:
                groups.setdefault(share, []).append(p)
            elif not p.get("fixed"):
                free_rows.append(p)
        self.variables: list[dict] = []
        for p in free_rows:
            self.variables.append(self._var([p]))
        for share, members in groups.items():
            kinds = {p["kind"] for p in members}
            if len(kinds) > 1:
                raise ValueError(f"共享组 {share} 跨越元件种类: {sorted(kinds)}")
            for p in members:
                if p.get("fixed"):
                    raise ValueError(f"参数 {p['name']} 既固定又在共享组 {share} 中")
            lo = max(float(p["lower"]) for p in members)
            hi = min(float(p["upper"]) for p in members)
            if not lo < hi:
                raise ValueError(f"共享组 {share} 的成员上下界无公共区间")
            self.variables.append({
                "share": share,
                "members": members,
                "names": [p["name"] for p in members],
                "lower": lo,
                "upper": hi,
                "value": float(members[0]["value"]),
                "kind": members[0]["kind"],
                "param": members[0]["param"],
            })
        self.nvar = len(self.variables)

    @staticmethod
    def _var(members):
        p = members[0]
        return {
            "share": "",
            "members": members,
            "names": [p["name"]],
            "lower": float(p["lower"]),
            "upper": float(p["upper"]),
            "value": float(p["value"]),
            "kind": p["kind"],
            "param": p["param"],
        }

    def _sigmas(self):
        if self.weighting == "sigma":
            return self.sigma_re.copy(), self.sigma_im.copy()
        if self.weighting == "modulus":
            s = np.abs(self.z_obs)
            s = np.maximum(s, 1e-30)
            return s, s
        return np.ones_like(self.freqs), np.ones_like(self.freqs)

    def _params_from_x(self, x):
        params = {}
        for p in self.parameters:
            params[p["name"]] = float(p["value"]) if p.get("fixed") else None
        for var, value in zip(self.variables, x):
            for name in var["names"]:
                params[name] = float(value)
        return params

    def residual(self, x):
        params = self._params_from_x(x)
        z = impedance(self.node, params, self.freqs)
        sre, sim = self._sigmas()
        re = (z.real - self.z_obs.real) / sre
        im = (z.imag - self.z_obs.imag) / sim
        return np.concatenate([re, im])

    def model(self, x):
        return impedance(self.node, self._params_from_x(x), self.freqs)


def _to_t(x, lo, hi):
    u = (x - lo) / (hi - lo)
    u = min(1.0 - 1e-9, max(1e-9, u))
    return math.log(u / (1.0 - u))


def _to_x(t, lo, hi):
    t = min(SIGMOID_LIMIT, max(-SIGMOID_LIMIT, float(t)))
    return lo + (hi - lo) / (1.0 + math.exp(-t))


def _expert_starts(problem: FitProblem):
    """从数据估计 EIS 典型量，构造少量高质量起点（确定性，不依赖种子）。"""
    z, f = problem.z_obs, problem.freqs
    re = z.real
    rs_est = float(np.min(re)) if len(re) else 1.0
    rspan = max(float(np.max(re) - rs_est), 1e-6)
    # |Z| 最接近 (rs + rspan/2) 的频率近似特征频率 ω*。
    target = rs_est + 0.5 * rspan
    i_star = int(np.argmin(np.abs(np.abs(z) - target)))
    omega_star = float(2 * np.pi * max(f[i_star], 1e-12))
    names = [v["names"][0] for v in problem.variables]
    seeds = []

    def build(scales, nvals=(0.85, 0.95, 0.99)):
        for rscale, nval in zip(scales, nvals):
            x = np.empty(problem.nvar)
            for k, var in enumerate(problem.variables):
                pname = var["param"]
                lo, hi = var["lower"], var["upper"]
                if pname == "R":
                    val = rscale * (rspan / max(len(names), 1) ** 0)
                    if k == 0 or "Rs" in var["names"][0] or var["names"][0].startswith("Rs"):
                        val = rs_est
                    else:
                        val = rspan * rscale
                elif pname == "C":
                    val = 1.0 / (omega_star * max(rspan, 1e-9)) * (1.0 / rscale)
                elif pname == "Y0":
                    # Y0 ≈ 1/(R·ω*^{1-n})，与 C=1/(Rω*) 在 n=1 时一致。
                    val = (1.0 / max(rspan, 1e-9)) * omega_star ** (nval - 1.0)
                elif pname == "n":
                    val = nval
                elif pname == "L":
                    val = rspan / omega_star
                else:
                    val = var["value"]
                x[k] = min(hi * (1 - 1e-7) if hi > 0 else hi - 1e-12,
                           max(lo + (hi - lo) * 1e-7, val))
            seeds.append(x)
    build((0.6, 1.0, 1.6))
    return seeds


def _initial_grid(problem: FitProblem, n_starts: int, seed: int):
    """多起点：数据启发专家点 + 当前值 + 分层拉丁超立方（全部确定性）。"""
    rng = np.random.default_rng(seed)
    current = np.array(
        [min(v["upper"] - 1e-12, max(v["lower"] + 1e-12, v["value"]))
         for v in problem.variables])
    starts = _expert_starts(problem) + [current]
    # 去重/限数：专家点优先
    unique = []
    seen = set()
    for x in starts:
        key = tuple(np.round(x, 8))
        if key not in seen:
            seen.add(key); unique.append(x)
    starts = unique[:n_starts]
    remain = n_starts - len(starts)
    if remain > 0:
        u = np.empty((remain, problem.nvar))
        for j in range(problem.nvar):
            perm = rng.permutation(remain)
            u[:, j] = (perm + rng.uniform(0.0, 1.0, remain)) / remain
        for i in range(remain):
            row = np.empty(problem.nvar)
            for j, var in enumerate(problem.variables):
                lo, hi, span = var["lower"], var["upper"], var["upper"] - var["lower"]
                ui = float(u[i, j])
                xval = lo * (hi / lo) ** ui if (lo > 0 and hi / lo > 1e6) else lo + span * ui
                row[j] = min(hi - span * 1e-9, max(lo + span * 1e-9, xval))
            starts.append(row)
    return np.array(starts[:n_starts])


def _t_to_x_all(problem, t):
    return np.array([
        _to_x(t[k], problem.variables[k]["lower"], problem.variables[k]["upper"])
        for k in range(problem.nvar)])


def _residual_t(problem, t):
    return problem.residual(_t_to_x_all(problem, t))


def _jacobian_t(problem: FitProblem, t, r0, h=1e-6):
    """t（logit）空间中心差分，自动反映有界映射的全部曲率。"""
    n = problem.nvar
    cols = []
    for j in range(n):
        tup = t.copy(); tup[j] += h
        tdn = t.copy(); tdn[j] -= h
        col = (_residual_t(problem, tup) - _residual_t(problem, tdn)) / (2 * h)
        if not np.all(np.isfinite(col)):
            tup2 = t.copy(); tup2[j] += h * 10
            col = (_residual_t(problem, tup2) - r0) / (h * 10)
        cols.append(col)
    return np.column_stack(cols)


def _jacobian(problem: FitProblem, x, r0):
    """物理量纲中心差分（仅用于协方差诊断），带数值失败回退。"""
    n = problem.nvar
    cols = []
    for j, var in enumerate(problem.variables):
        lo, hi = var["lower"], var["upper"]
        xj = float(x[j])
        h = max(abs(xj) * 1e-7, (hi - lo) * 1e-7, 1e-12)
        xp = min(hi - (hi - lo) * 1e-12, xj + h)
        xm = max(lo + (hi - lo) * 1e-12, xj - h)
        xup = x.copy(); xup[j] = xp
        xdn = x.copy(); xdn[j] = xm
        col = (problem.residual(xup) - problem.residual(xdn)) / max(xp - xm, 1e-30)
        if not np.all(np.isfinite(col)):
            xup2 = x.copy(); xup2[j] = min(hi - (hi - lo) * 1e-9, xj + (hi - lo) * 1e-4)
            col = (problem.residual(xup2) - r0) / max(xup2[j] - xj, 1e-30)
        cols.append(col)
    return np.column_stack(cols)


def _lm_once(problem: FitProblem, x0, max_iters=MAX_ITERS):
    """无界 t(logit) 空间上的 Levenberg-Marquardt。"""
    t = np.array([_to_t(x0[k], problem.variables[k]["lower"],
                        problem.variables[k]["upper"])
                  for k in range(problem.nvar)])
    r = _residual_t(problem, t)
    if not np.all(np.isfinite(r)):
        return x0, np.inf, "failed", 0
    chi2 = float(r @ r)
    lam = 1e-3
    radius = 5.0  # logit 步长信任域初始半径
    status = "max_iter"
    iters = 0
    for iters in range(1, max_iters + 1):
        jac = _jacobian_t(problem, t, r)
        jtj = jac.T @ jac
        g = jac.T @ r
        scale = np.maximum(np.diag(jtj), 1e-12)
        gnorm = float(np.max(np.abs(g) / scale))
        if gnorm < 1e-8:
            status = "converged"
            break
        accepted = False
        delta = None
        for _ in range(40):
            with np.errstate(over="ignore", invalid="ignore"):
                try:
                    delta = -np.linalg.solve(
                        jtj + lam * np.diag(scale), g)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
            if not np.all(np.isfinite(delta)):
                lam *= 10.0
                continue
            dnorm = float(np.linalg.norm(delta))
            if dnorm > radius:
                delta = delta * (radius / dnorm)
            tnew = np.clip(t + delta, -SIGMOID_LIMIT, SIGMOID_LIMIT)
            rnew = _residual_t(problem, tnew)
            if np.all(np.isfinite(rnew)):
                chi2_new = float(rnew @ rnew)
                if chi2_new < chi2:
                    stepnorm = float(np.max(np.abs(delta)))
                    rel = (chi2 - chi2_new) / max(chi2, 1e-30)
                    t, r, chi2 = tnew, rnew, chi2_new
                    lam = max(lam * 0.3, 1e-14)
                    radius = min(20.0, radius * (1.5 if dnorm >= radius*0.9 else 1.0))
                    accepted = True
                    if rel < 1e-12 and stepnorm < 1e-7:
                        status = "converged"
                    break
            lam *= 4.0
        if status == "converged":
            break
        if not accepted:
            radius = max(0.2, radius * 0.5)
            if radius <= 0.2 and gnorm > 1e-4:
                status = "stalled"
                break
    x = _t_to_x_all(problem, t)
    return x, chi2, status, iters


def _at_bound(x, var):
    """贴边界判定：与边界的相对距离按“值与跨度中的较大者”归一。

    绝宽区间（如 R∈[1e-3,1e6]）下，10Ω 的规范化分数虽然只有 1e-8，
    但它显然没有贴下边界；因此这里用 max(|x|, span*1e-6) 作尺度，
    只把真正被边界挡住的优化标记为贴界。
    """
    lo, hi = float(var["lower"]), float(var["upper"])
    span = hi - lo
    scale = max(abs(x), span * 1e-6)
    at_lo = (x - lo) <= BOUND_FRAC * scale
    at_hi = (hi - x) <= BOUND_FRAC * scale
    return at_lo, at_hi


@dataclass
class FitResult:
    chi2: float
    dof: int
    redchi2: float
    rms: float
    status: str
    iterations: int
    values: dict[str, float]
    variable_names: list[str]
    se: dict[str, float | None]
    correlation: dict[tuple[str, str], float]
    rank: int
    nvar: int
    condition: float | None
    near_singular: bool
    null_directions: list[list[float]]
    at_lower: list[str]
    at_upper: list[str]
    starts: list[dict] = field(default_factory=list)
    best_index: int = 0
    weighting: str = "unit"
    residual_re: list[float] = field(default_factory=list)
    residual_im: list[float] = field(default_factory=list)
    z_model: np.ndarray = None


def solve(problem: FitProblem, n_starts: int = 12, seed: int = 20240922) -> FitResult:
    x_grid = _initial_grid(problem, n_starts, seed)
    starts: list[dict] = []
    best = None
    for i, xstart in enumerate(x_grid):
        x, chi2, raw_status, iters = _lm_once(problem, xstart)
        at_lo, at_hi = [], []
        for k, var in enumerate(problem.variables):
            lo, hi = _at_bound(x[k], var)
            for name in var["names"]:
                if lo:
                    at_lo.append(name)
                if hi:
                    at_hi.append(name)
        if raw_status == "converged" and (at_lo or at_hi):
            status = "bound_only"
        else:
            status = raw_status
        rec = {
            "index": i,
            "initial": xstart.tolist(),
            "chi2": None if not math.isfinite(chi2) else chi2,
            "status": status,
            "iterations": iters,
            "x": x.tolist(),
            "at_lower": at_lo,
            "at_upper": at_hi,
        }
        starts.append(rec)
        if math.isfinite(chi2) and (best is None or chi2 < best[0]):
            best = (chi2, x, status, iters, i, at_lo, at_hi)
    chi2, x, status, iters, best_i, at_lo, at_hi = best
    r = problem.residual(x)
    dof = r.size - problem.nvar
    diag, z_model = _diagnostics(problem, x, r)
    values = problem._params_from_x(x)
    return FitResult(
        chi2=chi2, dof=dof, redchi2=chi2 / max(dof, 1),
        rms=float(math.sqrt(chi2 / r.size)),
        status=status, iterations=iters, values=values,
        variable_names=diag["names"], se=diag["se"],
        correlation=diag["correlation"], rank=diag["rank"],
        nvar=problem.nvar, condition=diag["condition"],
        near_singular=diag["near_singular"],
        null_directions=diag["null_directions"],
        at_lower=sorted(set(at_lo)), at_upper=sorted(set(at_hi)),
        starts=starts, best_index=best_i, weighting=problem.weighting,
        residual_re=r[:len(problem.freqs)].tolist(),
        residual_im=r[len(problem.freqs):].tolist(),
        z_model=z_model,
    )


def _diagnostics(problem: FitProblem, x, r):
    n = problem.nvar
    jac = _jacobian(problem, x, r)
    spans = np.array([v["upper"] - v["lower"] for v in problem.variables])
    names = [v["names"][0] if not v["share"] else f"共享:{v['share']}({'+'.join(v['names'])})"
             for v in problem.variables]
    ju = jac * spans[np.newaxis, :]
    try:
        u, s, vt = np.linalg.svd(ju, full_matrices=False)
    except np.linalg.LinAlgError:
        s = np.array([0.0])
    smax = s[0] if s.size else 0.0
    if smax <= 0 or not np.isfinite(smax):
        rank = 0
        cond = None
    else:
        rank = int(np.sum(s > SVD_RANK_CUTOFF * smax))
        s_min = s[rank - 1] if rank else smax
        cond = float(smax / max(s_min, np.finfo(float).tiny))
    near_singular = rank < n
    redchi2 = float(r @ r) / max(r.size - n, 1)

    se: dict[str, float | None] = {}
    correlation: dict[tuple[str, str], float] = {}
    null_dirs: list[list[float]] = []
    if not near_singular and n > 0:
        cov_u = redchi2 * (vt.T @ np.diag(1.0 / s ** 2) @ vt)
        std_u = np.sqrt(np.diag(cov_u))
        for k, var in enumerate(problem.variables):
            se_x = float(spans[k] * std_u[k])
            for nm in var["names"]:
                se[nm] = se_x
        d = np.outer(std_u, std_u)
        corr = cov_u / np.maximum(d, 1e-300)
        for a in range(n):
            for b in range(a + 1, n):
                correlation[(names[a], names[b])] = float(corr[a, b])
    else:
        for var in problem.variables:
            for nm in var["names"]:
                se[nm] = None
        for idx in range(rank, len(s)):
            null_dirs.append([float(v) for v in vt[idx]])
    for p in problem.parameters:
        if p.get("fixed"):
            se[p["name"]] = None
    return {
        "names": names, "se": se, "correlation": correlation,
        "rank": rank, "condition": cond,
        "near_singular": near_singular, "null_directions": null_dirs,
    }, problem.model(x)
