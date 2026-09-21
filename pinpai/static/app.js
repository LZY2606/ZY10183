"use strict";
const $ = (id) => document.getElementById(id);
let datasets = [], circuits = [], curRun = null, curCircuit = null, runs = [];
let activeTab = "nyquist";

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = await res.text();
    try { msg = JSON.parse(msg).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

async function boot() {
  datasets = await api("/api/datasets");
  circuits = await api("/api/circuits");
  fillSelect("dsSel", datasets, (d) => `#${d.id} ${d.name}`);
  fillSelect("cSel", circuits, (c) => `#${c.id} ${c.name}`);
  if (datasets.length) loadDataset();
  if (circuits.length) loadCircuit();
  await refreshRuns();
}
function fillSelect(id, items, label) {
  const el = $(id);
  el.innerHTML = "";
  items.forEach((it) => {
    const o = document.createElement("option");
    o.value = it.id; o.textContent = label(it);
    el.appendChild(o);
  });
}

function loadDataset() {
  const d = datasets.find((x) => x.id == $("dsSel").value);
  if (!d) return;
  const ex = d.excluded || [];
  $("dsInfo").innerHTML =
    `${d.freqs.length} 个正频率点，` +
    `${Math.min(...d.freqs).toExponential(1)}–${Math.max(...d.freqs).toExponential(1)} Hz` +
    (ex.length ? `<div class="warn">剔除 ${ex.length} 行（不进对数轴）：` +
      ex.map((e) => `原第 ${e.line} 行：${e.reason}`).join("；") + "</div>"
      : '<div class="ok">无被剔除行</div>') +
    (d.note ? `<div class="muted">${esc(d.note)}</div>` : "");
}
async function uploadDataset() {
  try {
    const r = await api("/api/datasets", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: $("dsName").value || "未命名",
                             csv_text: $("dsCsv").value }),
    });
    $("dsMsg").innerHTML =
      `<span class="ok">已导入 #${r.id}；剔除行：${(r.excluded || [])
        .map((e) => "第" + e.line + "行").join("、") || "无"}</span>`;
    datasets = await api("/api/datasets");
    fillSelect("dsSel", datasets, (d) => `#${d.id} ${d.name}`);
    $("dsSel").value = r.id; loadDataset();
  } catch (e) { $("dsMsg").textContent = "导入失败：" + e.message; }
}

function loadCircuit() {
  curCircuit = circuits.find((x) => x.id == $("cSel").value);
  if (!curCircuit) return;
  $("cExpr").value = curCircuit.expr;
  $("cNote").textContent = curCircuit.note || "";
  renderParams(curCircuit.param_defs, false);
}
function renderParams(defs, editable) {
  const t = $("paramTable");
  t.innerHTML = "<tr><th>参数键</th><th>单位</th><th>下界</th><th>初值</th>" +
                "<th>上界</th><th>共享组</th></tr>";
  defs.forEach((d, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td class="mono">${esc(d.key)}</td><td>${esc(d.unit)}</td>` +
      `<td><input data-k="${d.key}" data-f="lo" value="${d.lo}" style="width:78px"></td>` +
      `<td><input data-k="${d.key}" data-f="value" value="${d.value}" style="width:78px"></td>` +
      `<td><input data-k="${d.key}" data-f="hi" value="${d.hi}" style="width:78px"></td>` +
      `<td><input data-k="${d.key}" data-f="group" value="${d.group || ""}" style="width:64px"></td>`;
    t.appendChild(tr);
  });
}
function collectParams(defs) {
  const byKey = Object.fromEntries(defs.map((d) => [d.key, { ...d }]));
  document.querySelectorAll("#paramTable input").forEach((inp) => {
    const k = inp.dataset.k, f = inp.dataset.f;
    if (f === "group") byKey[k].group = inp.value.trim() || null;
    else byKey[k][f] = parseFloat(inp.value);
  });
  return defs.map((d) => byKey[d.key]);
}
async function validateExpr(extra) {
  const body = { expr: $("cExpr").value };
  if (curCircuit) body.param_defs = collectParams(curCircuit.param_defs);
  try {
    const r = await api("/api/validate/expr", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("cExpr").value = r.canonical;
    const old = curCircuit ? curCircuit.param_defs : [];
    const merged = r.param_keys.map((k) => {
      const found = (extra || []).find((d) => d.key === k) ||
                    old.find((d) => d.key === k) ||
                    r.param_defs.find((d) => d.key === k);
      return found;
    });
    renderParams(merged, true);
    $("convNote").textContent =
      "规范式：" + r.canonical + "\n" +
      "CPE 指数 n 物理界 [0,1]；复幂取主值支；时间因子 e^(+jωt)，故无源阻抗 Im<0。";
    return { canonical: r.canonical, defs: merged };
  } catch (e) { $("convNote").textContent = "表达式错误：" + e.message; return null; }
}
async function saveCircuit() {
  const v = await validateExpr(); if (!v) return;
  const name = prompt("电路名称", curCircuit ? curCircuit.name : "新候选");
  if (!name) return;
  try {
    await api(`/api/circuits/${curCircuit.id}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, expr: v.canonical, param_defs: v.defs,
                             note: $("cNote").value || "" }),
    });
    circuits = await api("/api/circuits");
    fillSelect("cSel", circuits, (c) => `#${c.id} ${c.name}`);
    $("cSel").value = curCircuit.id; loadCircuit();
  } catch (e) { alert("保存失败：" + e.message); }
}
async function newCircuit() {
  const v = await validateExpr(); if (!v) return;
  const name = prompt("新候选名称", "候选 D");
  try {
    const r = await api("/api/circuits", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, expr: v.canonical, param_defs: v.defs, note: "" }),
    });
    circuits = await api("/api/circuits");
    fillSelect("cSel", circuits, (c) => `#${c.id} ${c.name}`);
    $("cSel").value = r.id; loadCircuit();
  } catch (e) { alert("失败：" + e.message); }
}

async function runFit() {
  const payload = {
    dataset_id: +$("dsSel").value, circuit_id: +$("cSel").value,
    weight_mode: $("wMode").value, gamma: parseFloat($("gamma").value),
    n_starts: +$("nStarts").value, seed: +$("seed").value,
  };
  try {
    const r = await api("/api/fit", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await showRun(r.id);
  } catch (e) { alert("拟合失败：" + e.message); }
}
async function refitFixed() {
  if (!curRun) return alert("请先选择父运行");
  const fixed = {};
  document.querySelectorAll(".fixchk:checked").forEach((c) => {
    fixed[c.dataset.k] = parseFloat(c.dataset.v);
  });
  const body = { parent_id: curRun.id, fixed,
                 n_starts: +$("nStarts").value, seed: +$("seed").value,
                 label: "固定 " + Object.keys(fixed).join(",") };
  const r = await api("/api/refit", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  try { await showRun(r.id); } catch (e) { alert("重跑失败：" + e.message); }
}
async function refitGamma() {
  if (!curRun) return alert("请先选择父运行");
  const r = await api("/api/refit", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ parent_id: curRun.id, fixed: {}, gamma: 0.5,
                           label: "频率权重 γ=0.5 子版本",
                           n_starts: +$("nStarts").value, seed: +$("seed").value }),
  });
  try { await showRun(r.id); } catch (e) { alert("重跑失败：" + e.message); }
}

async function refreshRuns() {
  runs = await api("/api/runs");
  const byParent = {};
  runs.forEach((r) => (byParent[r.parent_id || 0] ||= []).push(r));
  const box = $("runTree"); box.innerHTML = "";
  const roots = byParent[0] || [];
  const walk = (r, depth) => {
    const d = document.createElement("div");
    d.style.paddingLeft = 8 + depth * 16 + "px";
    if (curRun && r.id === curRun.id) d.className = "sel";
    d.innerHTML = `#${r.id} [电路${r.circuit_id}/数据${r.dataset_id}] ` +
      `${r.weight_mode} γ=${r.gamma} ${r.label ? "· " + esc(r.label) : ""}`;
    d.onclick = () => showRun(r.id);
    box.appendChild(d);
    (byParent[r.id] || []).forEach((ch) => walk(ch, depth + 1));
  };
  roots.forEach((r) => walk(r, 0));
}
async function showRun(id) {
  curRun = await api(`/api/runs/${id}`);
  const res = curRun.result;
  renderSummary(res);
  renderFixBox(res);
  drawAll(res);
  renderStarts(res);
  renderParams(res);
  renderDiag(res);
  await refreshRuns();
}

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function fmt(x, n = 4) {
  if (x == null || !isFinite(x)) return "—";
  if (x === 0) return "0";
  const a = Math.abs(x);
  return a < 1e-3 || a >= 1e4 ? x.toExponential(2) : x.toPrecision(n);
}
function reasonBadge(reason) {
  const map = { converged: "真正收敛", at_bound: "贴界（非收敛）", max_iter: "达迭代上限" };
  return `<span class="badge ${reason}">${map[reason] || reason}</span>`;
}

function renderSummary(res) {
  const atU = res.bound_states.filter((b) => b.at_upper).map((b) => b.key);
  const atL = res.bound_states.filter((b) => b.at_lower).map((b) => b.key);
  const d = res.diagnosis;
  $("summary").innerHTML = `
    <div class="row">
      <span class="chip mono">${esc(res.expr_canonical)}</span>
      ${reasonBadge(res.reason)}
      <span class="chip">SSE(加权)=<b>${fmt(res.sse)}</b></span>
      <span class="chip">迭代 ${res.iterations}</span>
      <span class="chip">最优起点 #${res.best_start}</span>
      <span class="chip">权重 ${res.weight.mode} γ=${res.weight.gamma}</span>
      ${res.fixed && Object.keys(res.fixed).length
        ? `<span class="chip warn">固定 ${esc(Object.keys(res.fixed).join(","))}</span>` : ""}
    </div>
    <div class="muted" style="margin-top:6px">
      贴上界：${atU.length ? `<span class="warn"><b>${atU.join(", ")}</b></span>` : "无"}；
      贴下界：${atL.length ? `<span class="warn"><b>${atL.join(", ")}</b></span>` : "无"}
      · 设计矩阵秩 ${d.rank}/${d.n_free}
      · 条件数 ${d.condition ? fmt(d.condition, 3) : "∞"}
      · ${d.near_rank_deficient
        ? '<span class="bad"><b>近秩亏：标准误不给出（拒绝伪精密）</b></span>'
        : '<span class="ok">满秩：给出标准误与相关矩阵</span>'}
    </div>`;
}

function renderFixBox(res) {
  const box = $("fixBox");
  box.innerHTML = res.param_defs.map((d) => {
    const v = res.values[d.key];
    return `<label style="margin-right:10px;white-space:nowrap">
      <input type="checkbox" class="fixchk" data-k="${esc(d.key)}" data-v="${v}">
      ${esc(d.key)}=${fmt(v)}</label>`;
  }).join("");
}

function renderParams(res) {
  const t = $("resParams");
  const se = res.diagnosis.se;
  t.innerHTML = "<tr><th>参数</th><th>值</th><th>标准误</th><th>下界</th>" +
                "<th>上界</th><th>状态</th></tr>" +
    res.param_defs.map((d) => {
      const st = res.bound_states.find((b) => b.key === d.key) || {};
      const state = st.at_upper ? '<span class="warn">贴上界</span>'
        : st.at_lower ? '<span class="warn">贴下界</span>'
        : '<span class="ok">界内</span>';
      const seTxt = se ? fmt(se[d.key])
        : '<span class="bad" title="设计矩阵近秩亏">不给</span>';
      return `<tr><td class="mono">${esc(d.key)} <span class="muted">[${esc(d.unit)}]</span></td>` +
             `<td>${fmt(res.values[d.key])}</td><td>${seTxt}</td>` +
             `<td>${fmt(d.lo)}</td><td>${fmt(d.hi)}</td><td>${state}</td></tr>`;
    }).join("");

  const corr = res.diagnosis.correlation;
  const ct = $("resCorr");
  if (!corr) {
    ct.innerHTML = '<tr><td class="bad">近秩亏：相关矩阵不可靠，不予显示。</td></tr>';
    return;
  }
  const keys = corr.keys, m = corr.matrix;
  ct.innerHTML = "<tr><th></th>" + keys.map((k) => `<th>${esc(k)}</th>`).join("") +
    "</tr>" + keys.map((k, i) =>
      "<tr><th>" + esc(k) + "</th>" + m[i].map((v) => {
        const c = Math.abs(v) > 0.9 ? "bad" : Math.abs(v) > 0.6 ? "warn" : "";
        return `<td class="${c}">${v.toFixed(2)}</td>`;
      }).join("") + "</tr>").join("");
}

function renderDiag(res) {
  const d = res.diagnosis;
  $("diagBox").innerHTML = `
    <div class="note">判据：对尺度归一后的设计矩阵 J（列=物理参数直接灵敏度）做 SVD。
机器精度秩 = ${d.rank}/${d.n_free}；条件数阈值 1e8：超过即视为“形式满秩、实际近秩亏”，
此时 Gauss–Newton 标准误会被噪声放大，<b>系统拒绝给出标准误</b>，只报告零空间方向。</div>
    <div>奇异值：<span class="mono">${d.singular_values.map((v) => fmt(v, 3)).join(" , ")}</span></div>
    <div style="margin-top:8px">弱/零参数方向（参数变化但拟合几乎不响应 = 不可辨识方向）：</div>
    <div class="mono note">${(d.null_directions || []).map((dir) =>
      JSON.stringify(dir)).join("\n") || "无（各方向均可辨识）"}</div>`;
}

function renderStarts(res) {
  const box = $("startsBox");
  box.innerHTML = `<table><tr><th>#</th><th>排名</th><th>SSE(加权)</th>
    <th>迭代</th><th>终止原因</th><th>贴上界</th><th>贴下界</th><th>参数</th></tr>` +
    res.starts.map((s) => {
      const params = Object.entries(s.values)
        .map(([k, v]) => `${k}=${fmt(v)}`).join(" ");
      return `<tr><td>${s.start}</td><td>${s.rank}</td>` +
             `<td>${fmt(s.sse)}${s.rank === 0 ? ' ★' : ""}</td>` +
             `<td>${s.iterations}</td><td>${reasonBadge(s.reason)}</td>` +
             `<td class="${s.at_upper.length ? "warn" : ""}">${s.at_upper.join(",")}</td>` +
             `<td class="${s.at_lower.length ? "warn" : ""}">${s.at_lower.join(",")}</td>` +
             `<td class="mono" style="text-align:left;font-size:11px">${esc(params)}</td></tr>`;
    }).join("") + "</table>";
}

function showTab(name, el) {
  activeTab = name;
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tabp").forEach((p) =>
    p.style.display = p.id === "tab-" + name ? "" : "none");
  if (curRun) drawAll(curRun.result);
}

// ----------------------------- Canvas 绘图 -------------------------------- //
function setupCv(id) {
  const cv = $(id);
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 600, h = cv.height;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0);
  return { ctx, w, h };
}
function axes(ctx, view, xlabel, ylabel, xlog, ylog) {
  const { x0, y0, pw, ph, xmin, xmax, ymin, ymax } = view;
  ctx.strokeStyle = "#2b3750"; ctx.fillStyle = "#93a0b8";
  ctx.lineWidth = 1; ctx.font = "10px monospace";
  for (let i = 0; i <= 4; i++) {
    const gx = x0 + (pw * i) / 4;
    ctx.beginPath(); ctx.moveTo(gx, y0); ctx.lineTo(gx, y0 + ph); ctx.stroke();
    const t = xlog ? xmin * (xmax / xmin) ** (i / 4) : xmin + (xmax - xmin) * i / 4;
    ctx.fillText(xlog ? t.toExponential(0) : t.toFixed(1), gx - 14, y0 + ph + 14);
  }
  for (let i = 0; i <= 4; i++) {
    const gy = y0 + (ph * i) / 4;
    ctx.beginPath(); ctx.moveTo(x0, gy); ctx.lineTo(x0 + pw, gy); ctx.stroke();
    const t = ylog ? ymax * (ymin / ymax) ** (i / 4) : ymax - (ymax - ymin) * i / 4;
    ctx.fillText(ylog ? t.toExponential(0) : t.toFixed(2), 4, gy + 3);
  }
  ctx.fillStyle = "#c4d2ea";
  ctx.fillText(xlabel, x0 + pw / 2 - 20, y0 + ph + 30);
  ctx.save(); ctx.translate(12, y0 + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylabel, -40, 0); ctx.restore();
}
function makeView(w, h, xs, ys, pad) {
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  return { x0: pad.l, y0: pad.t, pw: w - pad.l - pad.r, ph: h - pad.t - pad.b,
           xmin, xmax, ymin, ymax };
}
function linMap(v, v0, v1) {
  const map = (v - v0) / (v1 - v0);
  return Math.max(0, Math.min(1, map));
}
function drawSeries(ctx, view, pts, color, logx, logy, sym) {
  ctx.fillStyle = color; ctx.strokeStyle = color;
  pts.forEach((p, i) => {
    const tx = logx ? Math.log(p.x / view.xmin) / Math.log(view.xmax / view.xmin)
                    : linMap(p.x, view.xmin, view.xmax);
    const ty = logy ? Math.log(view.ymax / p.y) / Math.log(view.ymax / view.ymin)
                    : 1 - linMap(p.y, view.ymin, view.ymax);
    const x = view.x0 + tx * view.pw, y = view.y0 + ty * view.ph;
    if (p.line && i) { ctx.beginPath();
      const q = pts[i - 1];
      const qx = view.x0 + (logx
        ? Math.log(q.x / view.xmin) / Math.log(view.xmax / view.xmin)
        : linMap(q.x, view.xmin, view.xmax)) * view.pw;
      const qy = view.y0 + (logy
        ? Math.log(view.ymax / q.y) / Math.log(view.ymax / view.ymin)
        : 1 - linMap(q.y, view.ymin, view.ymax)) * view.ph;
      ctx.moveTo(qx, qy); ctx.lineTo(x, y); ctx.stroke(); }
    ctx.beginPath(); ctx.arc(x, y, sym || 2.5, 0, 7); ctx.fill();
  });
}

function drawAll(res) {
  const pts = res.points;
  // Nyquist: x=Re, y=-Im（EIS 习惯）
  if (activeTab === "nyquist" || activeTab === "bode") {
    const { ctx, w, h } = setupCv("cvNyq");
    const xs = pts.flatMap((p) => [p.zreal_data, p.zreal_fit]);
    const ys = pts.flatMap((p) => [-p.zimag_data, -p.zimag_fit]);
    const view = makeView(w, h, xs, ys, { l: 52, r: 16, t: 16, b: 34 });
    axes(ctx, view, "Re(Z) [Ω]", "-Im(Z) [Ω]", false, false);
    drawSeries(ctx, view, pts.map((p) => ({ x: p.zreal_data, y: -p.zimag_data })),
               "#93a0b8", false, false, 3);
    const line = pts.slice().sort((a, b) => a.zreal_fit - b.zreal_fit)
      .map((p) => ({ x: p.zreal_fit, y: -p.zimag_fit, line: true }));
    drawSeries(ctx, view, line, "#6db1ff", false, false, 1.5);
  }
  if (activeTab === "bode") {
    drawBodeMag(res); drawBodePhase(res);
  }
  if (activeTab === "reim") { drawReImF(res, "cvReF", true);
                               drawReImF(res, "cvImF", false); }
  if (activeTab === "resid") drawResid(res);
}
function drawReImF(res, canvasId, isReal) {
  const { ctx, w, h } = setupCv(canvasId);
  const pts = res.points;
  const xs = pts.map((p) => p.freq_hz);
  const ys = pts.flatMap((p) => isReal
    ? [p.zreal_data, p.zreal_fit] : [p.zimag_data, p.zimag_fit]);
  const view = makeView(w, h, [1], ys, { l: 56, r: 16, t: 16, b: 34 });
  view.xmin = Math.min(...xs); view.xmax = Math.max(...xs);
  axes(ctx, view, "f [Hz]", isReal ? "Re(Z) [Ω]" : "Im(Z) [Ω]", true, false);
  if (!isReal) {
    ctx.strokeStyle = "#35415c";
    const zy = view.y0 + (1 - linMap(0, view.ymin, view.ymax)) * view.ph;
    ctx.beginPath(); ctx.moveTo(view.x0, zy);
    ctx.lineTo(view.x0 + view.pw, zy); ctx.stroke();
  }
  drawSeries(ctx, view, pts.map((p) =>
    ({ x: p.freq_hz, y: isReal ? p.zreal_data : p.zimag_data })),
    "#93a0b8", true, false, 3);
  drawSeries(ctx, view, pts.map((p) =>
    ({ x: p.freq_hz, y: isReal ? p.zreal_fit : p.zimag_fit, line: true })),
    "#6db1ff", true, false, 1.5);
}
function drawBodeMag(res) {
  const { ctx, w, h } = setupCv("cvMag");
  const pts = res.points;
  const mags = (zr, zi) => Math.hypot(zr, zi);
  const xs = pts.map((p) => p.freq_hz);
  const ys = pts.flatMap((p) => [mags(p.zreal_data, p.zimag_data),
                                 mags(p.zreal_fit, p.zimag_fit)]);
  const view = makeView(w, h, [Math.min(...xs)], ys,
    { l: 52, r: 16, t: 16, b: 34 });
  view.xmin = Math.min(...xs); view.xmax = Math.max(...xs);
  axes(ctx, view, "f [Hz]", "|Z| [Ω]", true, false);
  drawSeries(ctx, view, pts.map((p) =>
    ({ x: p.freq_hz, y: mags(p.zreal_data, p.zimag_data) })), "#93a0b8", true, false, 3);
  drawSeries(ctx, view, pts.map((p) =>
    ({ x: p.freq_hz, y: mags(p.zreal_fit, p.zimag_fit), line: true })),
    "#6db1ff", true, false, 1.5);
}
function drawBodePhase(res) {
  const { ctx, w, h } = setupCv("cvPhase");
  const pts = res.points;
  const ph = (zr, zi) => Math.atan2(zi, zr) * 180 / Math.PI;
  const xs = pts.map((p) => p.freq_hz);
  const ys = pts.flatMap((p) => [ph(p.zreal_data, p.zimag_data),
                                 ph(p.zreal_fit, p.zimag_fit)]);
  const view = makeView(w, h, [1], ys, { l: 52, r: 16, t: 16, b: 34 });
  view.xmin = Math.min(...xs); view.xmax = Math.max(...xs);
  axes(ctx, view, "f [Hz]", "φ [°]", true, false);
  ctx.strokeStyle = "#35415c";
  const zy = view.y0 + (1 - linMap(0, view.ymin, view.ymax)) * view.ph;
  ctx.beginPath(); ctx.moveTo(view.x0, zy); ctx.lineTo(view.x0 + view.pw, zy); ctx.stroke();
  drawSeries(ctx, view, pts.map((p) =>
    ({ x: p.freq_hz, y: ph(p.zreal_data, p.zimag_data) })), "#93a0b8", true, false, 3);
  drawSeries(ctx, view, pts.map((p) =>
    ({ x: p.freq_hz, y: ph(p.zreal_fit, p.zimag_fit), line: true })),
    "#6db1ff", true, false, 1.5);
}
function drawResid(res) {
  const { ctx, w, h } = setupCv("cvRes");
  const pts = res.points;
  const xs = pts.map((p) => p.freq_hz);
  const ys = pts.flatMap((p) => [p.resid_real, p.resid_imag]);
  const view = makeView(w, h, [1], ys, { l: 52, r: 16, t: 16, b: 34 });
  view.xmin = Math.min(...xs); view.xmax = Math.max(...xs);
  axes(ctx, view, "f [Hz]", "加权残差 √w·ΔZ", true, false);
  ctx.strokeStyle = "#35415c";
  const zy = view.y0 + (1 - linMap(0, view.ymin, view.ymax)) * view.ph;
  ctx.beginPath(); ctx.moveTo(view.x0, zy); ctx.lineTo(view.x0 + view.pw, zy); ctx.stroke();
  drawSeries(ctx, view, pts.map((p) =>
    ({ x: p.freq_hz, y: p.resid_real })), "#6db1ff", true, false, 3);
  drawSeries(ctx, view, pts.map((p) =>
    ({ x: p.freq_hz, y: p.resid_imag, line: true })), "#ffb454", true, false, 3);
}

async function exportAll() {
  const b = await api("/api/export");
  const blob = new Blob([JSON.stringify(b, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "pinpai-runs-" + new Date().toISOString().slice(0, 19) + ".json";
  a.click();
}
async function importAll(file) {
  if (!file) return;
  const text = await file.text();
  try {
    const r = await api("/api/import", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: text,
    });
    alert("导入完成：" + JSON.stringify(r.imported));
    await boot();
  } catch (e) { alert("导入失败：" + e.message); }
}
async function resetDb() {
  if (!confirm("将清空数据库并恢复内置 fixture，继续？")) return;
  await api("/api/reset", { method: "POST" });
  await boot();
}
window.addEventListener("resize", () => { if (curRun) drawAll(curRun.result); });
boot();
