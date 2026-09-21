"use strict";
// 频弧辨证台前端（零依赖，canvas 手绘 Nyquist/Bode/残差图）
const $ = (s, el=document) => el.querySelector(s);
const $$ = (s, el=document) => [...el.querySelectorAll(s)];
const api = async (path, opts={}) => {
  const res = await fetch(path, {
    headers: {"Content-Type": "application/json"}, ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
};
const fmt = (x, n=4) => {
  if (x === null || x === undefined || Number.isNaN(+x)) return "—";
  const prec = Math.min(8, Math.max(1, Math.round(n) || 4));
  x = +x;
  if (Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0)) return x.toExponential(2);
  return x.toPrecision(prec);
};

const state = {
  datasets: [], circuits: [], selectedDs: null, selectedC: null,
  previewParams: null, lastFitRuns: [],
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  $$(".tab").forEach(t => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  $("#btn-new-ds").addEventListener("click", createDataset);
  $("#btn-preview").addEventListener("click", previewCircuit);
  $("#btn-new-c").addEventListener("click", createCircuit);
  $("#btn-save-params").addEventListener("click", saveParams);
  $("#btn-fit-all").addEventListener("click", fitAll);
  $("#btn-export").addEventListener("click", () => location.href = "/api/export/download");
  $("#btn-reset").addEventListener("click", resetDb);
  $("#file-import").addEventListener("change", importFile);
  $("#modal-close").addEventListener("click", () => $("#modal").classList.add("hidden"));
  await loadCatalog();
  await refreshAll();
}

function switchTab(name) {
  $$(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  $$(".panel").forEach(p => p.classList.toggle("active", p.id === `tab-${name}`));
  if (name === "runs") loadTree();
}

async function refreshAll() {
  state.datasets = await api("/api/datasets");
  state.circuits = await api("/api/circuits");
  if (!state.selectedDs && state.datasets[0]) state.selectedDs = state.datasets[0].id;
  renderDatasets(); renderDatasetDetail(); renderDatasetSelect();
  renderCircuits(); renderParamTable();
}

async function createDataset() {
  const name = $("#new-ds-name").value.trim();
  const text = $("#new-ds-text").value;
  const box = $("#ds-parse-msg");
  try {
    const r = await api("/api/datasets", {method: "POST", body: {name, text}});
    box.innerHTML = `<span class="infobox">已入库：有效 ${r.valid} 行；排除 ${r.excluded.length} 行（原始行号：${r.excluded.map(e=>e.line).join(", ")||"无"}）</span>`;
    await refreshAll();
  } catch (e) { box.innerHTML = `<span class="warnbox">${e.message}</span>`; }
}

function renderDatasets() {
  $("#dataset-list").innerHTML = state.datasets.map(d => `
    <div class="item ${d.id===state.selectedDs?"sel":""}" data-id="${d.id}">
      <div><b>${esc(d.name)}</b>
        <div class="dim">有效 ${d.n_ok} 行 · 排除 ${d.n_excluded} 行 · ${esc(d.source||"")}</div></div>
      <span class="dim">${new Date(d.created_at*1000).toLocaleString()}</span>
    </div>`).join("");
  $$("#dataset-list .item").forEach(el => el.addEventListener("click", () => {
    state.selectedDs = el.dataset.id; renderDatasets(); renderDatasetDetail();
  }));
}

function renderDatasetSelect() {
  $("#fit-dataset").innerHTML = state.datasets.map(d =>
    `<option value="${d.id}" ${d.id===state.selectedDs?"selected":""}>${esc(d.name)}</option>`).join("");
}

async function renderDatasetDetail() {
  if (!state.selectedDs) return;
  const d = await api(`/api/datasets/${state.selectedDs}`);
  $("#ds-title").textContent = d.name;
  const excluded = d.points.filter(p => p.status !== "ok");
  $("#excluded-box").innerHTML = excluded.length ?
    `<div class="warnbox">以下 ${excluded.length} 行不进入对数轴与拟合：` +
    excluded.map(p => `物理行 ${p.line_no}（数据序号 ${p.row_num}）f=${p.freq}：${esc(p.reason)}`).join("；") + "</div>"
    : `<div class="infobox">没有被排除的行。</div>`;
  $("#points-table").innerHTML =
    `<tr><th>物理行</th><th>序号</th><th>f / Hz</th><th>Z′ / Ω</th><th>Z″ / Ω</th><th>σre</th><th>σim</th><th>状态</th></tr>` +
    d.points.map(p => `<tr class="${p.status!=="ok"?"excl":""}">
      <td>${p.line_no}</td><td>${p.row_num}</td><td>${p.freq===null?"—":fmt(p.freq)}</td>
      <td>${p.z_re===null?"—":fmt(p.z_re)}</td><td>${p.z_im===null?"—":fmt(p.z_im)}</td>
      <td>${fmt(p.sigma_re)}</td><td>${fmt(p.sigma_im)}</td>
      <td>${p.status==="ok"?"✓":esc(p.reason)}</td></tr>`).join("");
}

function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}

async function previewCircuit() {
  const text = $("#new-c-text").value.trim();
  const msg = $("#circuit-msg");
  try {
    const r = await api("/api/circuits/preview", {method:"POST", body:{name:"x", text}});
    state.previewParams = r.parameters;
    msg.innerHTML = `<span class="infobox">稳定表达：<code>${esc(r.canonical)}</code>；${r.parameters.length} 个参数，确认后保存候选。</span>`;
    renderPreviewParamTable(r.parameters);
  } catch (e) { msg.innerHTML = `<span class="warnbox">${esc(e.message)}</span>`; }
}

function renderPreviewParamTable(params) {
  // 临时在参数表里编辑（保存候选时一起提交）
  state.previewParams = params;
  if (!state.selectedC) renderParamTable();
}

async function createCircuit() {
  const name = $("#new-c-name").value.trim();
  const text = $("#new-c-text").value.trim();
  const msg = $("#circuit-msg");
  try {
    const r = await api("/api/circuits", {method:"POST",
      body:{name, text, parameters: state.previewParams || undefined}});
    msg.innerHTML = `<span class="infobox">已保存候选 <code>${esc(r.canonical)}</code></span>`;
    state.previewParams = null;
    await refreshAll();
    const c = state.circuits.find(x => x.canonical === r.canonical && x.name === name);
    if (c) { state.selectedC = c.id; renderCircuits(); renderParamTable(); }
  } catch (e) { msg.innerHTML = `<span class="warnbox">${esc(e.message)}</span>`; }
}

function renderCircuits() {
  $("#circuit-list").innerHTML = state.circuits.map(c => `
    <div class="item ${c.id===state.selectedC?"sel":""}" data-id="${c.id}">
      <div><b>${esc(c.name)}</b> <code>${esc(c.canonical)}</code></div>
      <span class="dim">${new Date(c.created_at*1000).toLocaleString()}</span>
    </div>`).join("");
  $$("#circuit-list .item").forEach(el => el.addEventListener("click", () => {
    state.selectedC = el.dataset.id; renderCircuits(); renderParamTable();
  }));
}

function renderParamTable() {
  const tbl = $("#param-table");
  let params = null, editable = false;
  if (state.previewParams && !state.selectedC) {
    params = state.previewParams; editable = true;
  } else {
    const c = state.circuits.find(x => x.id === state.selectedC);
    if (c) params = c.parameters;
  }
  $("#btn-save-params").style.display = state.selectedC ? "" : "none";
  if (!params) { tbl.innerHTML = `<tr><td class="dim">选择一个候选或在左侧“新建候选”后点解析/预览。</td></tr>`; return; }
  tbl.innerHTML = `<tr><th>参数</th><th>元件</th><th>单位</th><th>当前值</th><th>下界</th><th>上界</th><th>固定</th><th>共享组</th></tr>` +
    params.map((p,i) => `<tr data-i="${i}" data-name="${esc(p.name)}">
      <td><code>${esc(p.name)}</code></td><td>${esc(p.element)}</td><td>${esc(p.unit)}</td>
      <td><input class="pv" value="${p.value}"></td>
      <td><input class="pl" value="${p.lower}"></td>
      <td><input class="pu" value="${p.upper}"></td>
      <td><input class="pf" type="checkbox" ${p.fixed?"checked":""}></td>
      <td><input class="ps" value="${esc(p.share||"")}" placeholder="如 g1" style="width:80px"></td></tr>`).join("");
}

async function saveParams() {
  if (!state.selectedC) return;
  const params = $$("#param-table tr[data-name]").map(tr => {
    const c = state.circuits.find(x => x.id === state.selectedC);
    const old = c.parameters.find(p => p.name === tr.dataset.name);
    return {
      ...old,
      value: parseFloat($(".pv",tr).value),
      lower: parseFloat($(".pl",tr).value),
      upper: parseFloat($(".pu",tr).value),
      fixed: $(".pf",tr).checked,
      share: $(".ps",tr).value.trim(),
    };
  });
  try {
    await api(`/api/circuits/${state.selectedC}`, {method:"PATCH",
      body:{parameters: params}});
    $("#param-msg").textContent = "已保存";
    await refreshAll();
  } catch (e) { $("#param-msg").innerHTML = `<span class="warnbox">${esc(e.message)}</span>`; }
}

// ---------------- canvas 绘图 ----------------
function makeChart(canvas, series, opts={}) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 400, H = canvas.clientHeight || 190;
  canvas.width = W*dpr; canvas.height = H*dpr;
  const ctx = canvas.getContext("2d"); ctx.scale(dpr,dpr);
  ctx.fillStyle = "#0c1322"; ctx.fillRect(0,0,W,H);
  const m = {l:46,r:12,t:12,b:30};
  const all = series.flatMap(s => s.data);
  if (!all.length) return;
  let xs = all.map(p=>p[0]), ys = all.map(p=>p[1]);
  const xlog = opts.xlog, ylog = opts.ylog;
  let [xmin,xmax] = (opts.xrange||[]).length ? opts.xrange : [Math.min(...xs),Math.max(...xs)];
  let [ymin,ymax] = (opts.yrange||[]).length ? opts.yrange : [Math.min(...ys),Math.max(...ys)];
  if (xmin===xmax) {xmin*=0.9;xmax*=1.1} if (ymin===ymax){ymin-=1;ymax+=1}
  const tx = x => xlog ? m.l+(Math.log10(x)-Math.log10(xmin))/(Math.log10(xmax)-Math.log10(xmin))*(W-m.l-m.r)
                       : m.l+(x-xmin)/(xmax-xmin)*(W-m.l-m.r);
  const ty = y => ylog ? H-m.b-(Math.log10(y)-Math.log10(ymin))/(Math.log10(ymax)-Math.log10(ymin))*(H-m.t-m.b)
                       : H-m.b-(y-ymin)/(ymax-ymin)*(H-m.t-m.b);
  ctx.strokeStyle="#28355"; ctx.strokeStyle="#28355a"; ctx.lineWidth=1;
  ctx.strokeRect(m.l,m.t,W-m.l-m.r,H-m.t-m.b);
  ctx.fillStyle="#8b97b3"; ctx.font="10px sans-serif";
  for (let i=0;i<=4;i++){
    const xv = xlog ? xmin*Math.pow(xmax/xmin,i/4) : xmin+(xmax-xmin)*i/4;
    const yv = ymin+(ymax-ymin)*i/4;
    ctx.fillText(xv.toExponential(1), m.l+(W-m.l-m.r)*i/4-18, H-m.b+14);
    ctx.fillText((yv>=1000?yv.toExponential(1):(+yv).toPrecision(2)), 2, H-m.b-(H-m.t-m.b)*i/4+3);
  }
  ctx.fillStyle="#8b97b3";
  ctx.fillText(opts.xlabel||"", W-m.r-60, H-4);
  ctx.save(); ctx.translate(11,m.t+10); ctx.rotate(-Math.PI/2);
  ctx.fillText(opts.ylabel||"",0,0); ctx.restore();
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.fillStyle = s.color;
    if (s.line !== false) {
      ctx.beginPath();
      s.data.forEach((p,i)=>{ const X=tx(p[0]),Y=ty(p[1]); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y); });
      ctx.lineWidth=1.6; ctx.stroke();
    }
    if (s.dots) {
      s.data.forEach(p=>{ ctx.beginPath(); ctx.arc(tx(p[0]),ty(p[1]),2.2,0,7); ctx.fill(); });
    }
  }
}

// ---------------- 拟合 ----------------
async function fitAll() {
  const dataset_id = $("#fit-dataset").value;
  const weighting = $("#fit-weight").value;
  const n_starts = parseInt($("#fit-starts").value,10);
  const box = $("#fit-cards"); box.innerHTML = "";
  $("#fit-busy").textContent = "拟合中…";
  for (const c of state.circuits) {
    const card = document.createElement("div");
    card.className = "fitcard";
    card.innerHTML = `<h3>${esc(c.name)} <code>${esc(c.canonical)}</code></h3>
      <div class="dim">计算中…</div>`;
    box.appendChild(card);
    try {
      const r = await api("/api/fit", {method:"POST",
        body:{circuit_id:c.id, dataset_id, weighting, n_starts}});
      const view = await api(`/api/runs/${r.id}`);
      renderFitCard(card, view, {dataset_id});
    } catch(e) {
      card.querySelector(".dim").innerHTML = `<span class="warnbox">${esc(e.message)}</span>`;
    }
  }
  $("#fit-busy").textContent = "完成";
  setTimeout(()=>$("#fit-busy").textContent="", 3000);
}

function statusBadge(status) {
  const map = {converged:["good","真正收敛"], bound_only:["warn","贴边界·非收敛"],
    max_iter:["bad","迭代用尽"], stalled:["bad","停滞"], failed:["bad","失败"]};
  const [cls,label] = map[status] || ["",status];
  return `<span class="badge ${cls}">${label}</span>`;
}

function renderFitCard(card, view, ctx) {
  const r = view.run;
  const pts = view.points;
  const singular = r.near_singular;
  card.innerHTML = `
    <h3>${esc(view.circuit_name)} <code>${esc(view.canonical)}</code></h3>
    <div class="row" style="margin:4px 0">
      ${statusBadge(r.status)}
      <span class="badge ${singular?"warn":"good"}">数值秩 ${r.rank}/${r.nvar}</span>
      <span class="dim">χ²=${fmt(r.chi2,5)} · χ²/dof=${fmt(r.redchi2,4)} · 权重 ${r.weighting}</span>
    </div>
    ${r.at_upper.length?`<div class="warnbox">贴上界（优化想继续而被挡住，不是收敛）：${r.at_upper.map(x=>`<b>${esc(x)}</b>`).join("、")}</div>`:""}
    ${r.at_lower.length?`<div class="warnbox">贴下界：${r.at_lower.map(x=>`<b>${esc(x)}</b>`).join("、")}</div>`:""}
    ${singular?`<div class="warnbox">设计矩阵近秩亏（条件数≈${r.condition?fmt(r.condition,3):"∞"}）：<b>不报告标准误</b>，只给相关方向与零空间。</div>`
      :`<div class="infobox">满秩；标准误由 SVD 协方差给出（已乘 χ²/dof）。</div>`}
    <div class="plots2">
      <div><canvas data-p="ny"></canvas><div class="dim" style="text-align:center">Nyquist：−Z″ vs Z′</div></div>
      <div><canvas data-p="bode"></canvas><div class="dim" style="text-align:center">Bode 幅频（实线模型/点观测）</div></div>
      <div><canvas data-p="res"></canvas><div class="dim" style="text-align:center">加权残差 vs log f</div></div>
      <div><canvas data-p="phase"></canvas><div class="dim" style="text-align:center">Bode 相频 (°)</div></div>
    </div>
    <div><b class="dim">多起点（并排保留，蓝框=最优 χ²）：</b>
      <div class="starts">${r.starts.map((s,i)=>{
        const cls = ["converged","good","bound_only","warn","max_iter","bad","stalled","bad","failed","bad"];
        let c=""; for(let k=0;k<cls.length;k+=2) if(cls[k]===s.status) c=cls[k+1];
        return `<span class="dot ${i===r.starts.findIndex(x=>x.chi2===Math.min(...r.starts.filter(y=>y.chi2!==null).map(y=>y.chi2)))?'best':''}"
          style="border-color:var(--${c==='good'?'good':c==='warn'?'warn':'bad'})"
          data-idx="${i}" title="起点${i}: ${s.status}, χ²=${s.chi2===null?'∞':fmt(s.chi2,5)}">${i}:${s.status==='converged'?'✓':s.status==='bound_only'?'⇡':'…'}</span>`;
      }).join("")}</div></div>
    <div class="row">
      <button class="ghost btn-detail">完整详情（参数/SE/相关/零空间）</button>
      <button class="ghost btn-fix">固定参数重跑（子版本）</button>
      <button class="ghost btn-reweight">改权重生成子版本</button>
    </div>`;
  requestAnimationFrame(()=>drawCharts(card, pts));
  card.querySelector(".btn-detail").addEventListener("click",()=>openDetail(view));
  card.querySelector(".btn-fix").addEventListener("click",()=>openFix(view, ctx));
  card.querySelector(".btn-reweight").addEventListener("click",()=>openReweight(view, ctx));
}

function drawCharts(card, pts) {
  const f = p=>p.freq;
  const ny = card.querySelector('canvas[data-p="ny"]');
  const obsN = pts.map(p=>[p.re_o,-p.im_o]).sort((a,b)=>a[0]-b[0]);
  const modN = pts.map(p=>[p.re_m,-p.im_m]).sort((a,b)=>a[0]-b[0]);
  const xR = [Math.min(...obsN.map(p=>p[0]),...modN.map(p=>p[0]))-2,
               Math.max(...obsN.map(p=>p[0]),...modN.map(p=>p[0]))+2];
  const yR = [0, Math.max(...obsN.map(p=>p[1]),...modN.map(p=>p[1]))*1.08];
  makeChart(ny, [
    {data:modN,color:"#5aa9ff"},
    {data:obsN,color:"#ffb86b",dots:true,line:false},
  ],{xrange:xR,yrange:yR,xlabel:"Z′/Ω",ylabel:"−Z″/Ω"});

  const bode = card.querySelector('canvas[data-p="bode"]');
  makeChart(bode, [
    {data:pts.map(p=>[p.freq,p.mag_m]),color:"#5aa9ff"},
    {data:pts.map(p=>[p.freq,p.mag_o]),color:"#ffb86b",dots:true,line:false},
  ],{xlog:true,ylog:true,xlabel:"f/Hz",ylabel:"|Z|/Ω"});

  const res = card.querySelector('canvas[data-p="res"]');
  const rmin = Math.min(-2.5,...pts.map(p=>Math.min(p.res_re,p.res_im)));
  const rmax = Math.max(2.5,...pts.map(p=>Math.max(p.res_re,p.res_im)));
  makeChart(res, [
    {data:pts.map(p=>[p.freq,p.res_re]),color:"#3ecf8e",dots:true},
    {data:pts.map(p=>[p.freq,p.res_im]),color:"#b48cff",dots:true},
  ],{xlog:true,yrange:[rmin,rmax],xlabel:"f/Hz",ylabel:"r/σ"});

  const phase = card.querySelector('canvas[data-p="phase"]');
  makeChart(phase, [
    {data:pts.map(p=>[p.freq,p.phase_m]),color:"#5aa9ff"},
    {data:pts.map(p=>[p.freq,p.phase_o]),color:"#ffb86b",dots:true,line:false},
  ],{xlog:true,xlabel:"f/Hz",ylabel:"phase/°"});
}

function openDetail(view) {
  const r = view.run;
  $("#modal-title").textContent = `运行详情 · ${view.circuit_name}`;
  const corr = r.corr.length
    ? `<div class="corrgrid">${r.corr.map(c=>{
        const v=c.value; const col=Math.abs(v)>0.95?"var(--bad)":Math.abs(v)>0.8?"var(--warn)":"var(--good)";
        return `<div class="corr">${esc(c.a)} ↔ ${esc(c.b)}<br><b style="color:${col}">${fmt(v,4)}</b></div>`;
      }).join("")}</div>`
    : `<div class="dim">秩缺：相关矩阵不可用（下方给零空间）。</div>`;
  const nulls = r.null_directions && r.null_directions.length
    ? r.null_directions.map((d,i)=>`<div class="nulldir">v_${i+1} = [${d.map(x=>(+x).toFixed(3)).join(", ")}]</div>`).join("")
    : `<div class="dim">无数值零方向。</div>`;
  $("#modal-body").innerHTML = `
    <div class="two">
      <div>
        <h4>参数（SE 仅满秩时给出）</h4>
        <div class="table-wrap" style="max-height:38vh">
        <table><tr><th>参数</th><th>值</th><th>SE</th><th>单位</th><th>标记</th></tr>
        ${r.params.map(p=>`<tr><td><code>${esc(p.name)}</code></td>
          <td>${fmt(p.value,6)}</td><td>${p.se===null?'<span class="dim">不给伪精密 SE</span>':"±"+fmt(p.se,3)}</td>
          <td>${esc(p.unit)}</td><td>${p.fixed?'<span class="tag fix">固定</span>':""}
          ${r.at_upper.includes(p.name)?'<span class="tag up">贴上界</span>':""}
          ${r.at_lower.includes(p.name)?'<span class="tag lo">贴下界</span>':""}</td></tr>`).join("")}
        </table></div>
        <p class="dim">排除行（原始物理行号）：${view.excluded.map(e=>`行${e.line_no}: ${esc(e.reason)}`).join("；")||"无"}</p>
      </div>
      <div>
        <h4>参数相关（自由变量，区间标定后）</h4>${corr}
        <h4 style="margin-top:12px">零空间方向（近奇异组合，${r.rank}/${r.nvar}）</h4>${nulls}
        <h4 style="margin-top:12px">多起点全部结果</h4>
        <div class="table-wrap" style="max-height:26vh"><table>
        <tr><th>#</th><th>χ²</th><th>状态</th><th>迭代</th><th>贴界</th></tr>
        ${r.starts.map((s,i)=>`<tr><td>${i}${i===bestIdx(r)?' ★':''}</td>
          <td>${s.chi2===null?"∞":fmt(s.chi2,5)}</td><td>${esc(s.status)}</td>
          <td>${s.iterations}</td><td class="dim">[${[...s.at_upper,...s.at_lower].join(",")}]</td></tr>`).join("")}
        </table></div>
      </div>
    </div>`;
  $("#modal").classList.remove("hidden");
}

function bestIdx(r) {
  let bi=0,bv=Infinity;
  r.starts.forEach((s,i)=>{ if(s.chi2!==null && s.chi2<bv){bv=s.chi2;bi=i;} });
  return bi;
}

function openFix(view, ctx) {
  const r = view.run;
  $("#modal-title").textContent = "固定参数后重跑（父版本保留）";
  $("#modal-body").innerHTML = `
    <p class="dim">勾选要固定的参数并给出固定值（默认取父版本拟合值）；其余参数自由重跑。</p>
    <div class="table-wrap"><table><tr><th></th><th>参数</th><th>固定值</th></tr>
    ${r.params.map(p=>`<tr><td><input type="checkbox" class="fx" data-name="${esc(p.name)}"></td>
      <td><code>${esc(p.name)}</code></td>
      <td><input class="fv" data-name="${esc(p.name)}" value="${p.value}"></td></tr>`).join("")}
    </table></div>
    <div class="row" style="margin-top:10px">
      <label>权重 <select id="child-weight">
        <option value="sigma"${r.weighting==='sigma'?' selected':''}>σ 加权</option>
        <option value="unit"${r.weighting==='unit'?' selected':''}>等权</option>
        <option value="modulus"${r.weighting==='modulus'?' selected':''}>幅模</option></select></label>
      <label>起点 <input id="child-starts" type="number" value="${r.n_starts||12}" style="width:60px"></label>
      <button class="primary" id="do-fix">生成子版本并重跑</button>
      <span id="fix-msg"></span>
    </div>`;
  $("#modal").classList.remove("hidden");
  $("#do-fix").addEventListener("click", async () => {
    const fixed={};
    $$(".fx:checked").forEach(cb=>{ fixed[cb.dataset.name]=parseFloat($(`.fv[data-name="${cb.dataset.name}"]`).value); });
    if (!Object.keys(fixed).length){ $("#fix-msg").innerHTML='<span class="warnbox">至少勾选一个参数</span>'; return; }
    $("#fix-msg").textContent="计算中…";
    try {
      const res = await api("/api/fit/child",{method:"POST",body:{
        parent_id:r.id, action:"fix", fixed,
        weighting:$("#child-weight").value, n_starts:parseInt($("#child-starts").value,10),
        circuit_id:r.circuit_id, dataset_id:r.dataset_id}});
      $("#modal").classList.add("hidden");
      switchTab("runs"); await loadTree();
      const view2 = await api(`/api/runs/${res.id}`);
      openDetail(view2);
    } catch(e){ $("#fix-msg").innerHTML=`<span class="warnbox">${esc(e.message)}</span>`; }
  });
}

function openReweight(view, ctx) {
  const r = view.run;
  $("#modal-title").textContent = "修改频率权重 → 可追溯子版本";
  $("#modal-body").innerHTML = `
    <p class="dim">父版本 <code>${r.id}</code>（${r.weighting}）原样保留；使用新权重重跑并挂到父节点下。</p>
    <div class="row"><label>新权重 <select id="rw">
      <option value="sigma">σ 加权（数据列）</option>
      <option value="unit">等权 unit</option>
      <option value="modulus">幅模 modulus</option></select></label>
      <label>起点 <input id="rw-starts" type="number" value="${r.n_starts||12}" style="width:60px"></label>
      <button class="primary" id="do-rw">生成子版本</button><span id="rw-msg"></span></div>`;
  $("#modal").classList.remove("hidden");
  $("#do-rw").addEventListener("click", async ()=>{
    $("#rw-msg").textContent="计算中…";
    try{
      const res=await api("/api/fit/child",{method:"POST",body:{
        parent_id:r.id, action:"reweight",
        weighting:$("#rw").value, n_starts:parseInt($("#rw-starts").value,10),
        circuit_id:r.circuit_id, dataset_id:r.dataset_id}});
      $("#modal").classList.add("hidden");
      switchTab("runs"); await loadTree();
      openDetail(await api(`/api/runs/${res.id}`));
    }catch(e){$("#rw-msg").innerHTML=`<span class="warnbox">${esc(e.message)}</span>`;}
  });
}

// ---------------- 版本树 ----------------
async function loadTree() {
  const [tree, circuits, datasets] = await Promise.all(
    [api("/api/runs/tree"), api("/api/circuits"), api("/api/datasets")]);
  const cmap = Object.fromEntries(circuits.map(c=>[c.id,c]));
  const dmap = Object.fromEntries(datasets.map(d=>[d.id,d]));
  const byId = Object.fromEntries(tree.runs.map(r=>[r.id,r]));
  const roots = tree.runs.filter(r=>!r.parent_id);
  const box = $("#run-tree");
  if (!roots.length){ box.innerHTML = '<p class="dim">还没有运行。去“③ 拟合对照”跑一次。</p>'; return; }
  const children = pid => tree.runs.filter(r=>r.parent_id===pid);
  function nodeHtml(r, depth, isRoot){
    const c = cmap[r.circuit_id]||{}; const d = dmap[r.dataset_id]||{};
    const kids = children(r.id);
    const badge = statusBadge(r.status);
    const tags = (JSON.parse(r.at_upper||"[]")).map(x=>`<span class="tag up">↑${esc(x)}</span>`).join("")
      + (JSON.parse(r.at_lower||"[]")).map(x=>`<span class="tag lo">↓${esc(x)}</span>`).join("");
    const sing = r.near_singular ? `<span class="badge warn">秩缺 ${r.rank}/${r.nvar}</span>` : "";
    return `<div class="node ${isRoot?'root':''}" style="margin-left:${depth*18}px">
      <div class="nline">
        ${badge}${sing}
        <a href="#" class="run-link" data-id="${r.id}"><b>${esc(c.name||r.circuit_id)}</b></a>
        <code>${esc(c.canonical||"")}</code>
        <span class="dim">${esc(r.version_label||"")} · ${esc(r.weighting)} · χ²=${fmt(r.chi2,5)}</span>
        ${tags}
        ${r.fixed_note?`<span class="tag fix">${esc(r.fixed_note)}</span>`:""}
        <span class="dim">${new Date(r.created_at*1000).toLocaleString()}</span>
      </div>${kids.map(k=>nodeHtml(k,depth+1,false)).join("")}</div>`;
  }
  box.innerHTML = roots.map(r=>nodeHtml(r,0,true)).join("");
  $$("#run-tree .run-link").forEach(a=>a.addEventListener("click",async e=>{
    e.preventDefault(); openDetail(await api(`/api/runs/${a.dataset["id"]}`));
  }));
}

// ---------------- 约定 ----------------
async function loadCatalog() {
  const cat = await api("/api/catalog");
  const cv = cat.conventions;
  const rows = Object.entries(cv.weightings).map(([k,v])=>`<tr><td><code>${k}</code></td><td>${v}</td></tr>`).join("");
  const elrows = Object.values(cat.elements).map(e=>
    `<tr><td><b>${e.kind}</b> ${esc(e.cn)}</td><td>${e.params.join(", ")}</td>
     <td>${Object.entries(e.units).map(([k,v])=>`${k}:${v||"无量纲"}`).join("; ")}</td>
     <td>[${Object.values(e.lower).map(fmt).join(", ")}] – [${Object.values(e.upper).map(fmt).join(", ")}]</td></tr>`).join("");
  $("#conv-body").innerHTML = `
    <ul>
      <li><b>时间约定</b>：${cv.time}</li>
      <li><b>CPE 复幂主值支</b>：<code>${esc(cv.cpe)}</code></li>
      <li><b>Warburg</b>：<code>${esc(cv.warburg)}</code></li>
      <li><b>CPE 指数边界</b>：${esc(cv.n_bounds)}</li>
      <li><b>串并联文本</b>：${esc(cv.series_parallel)}</li>
      <li><b>非正频率</b>：${esc(cv.nonpositive_freq)}</li>
      <li><b>秩与 SE</b>：${esc(cv.rank)}</li>
      <li><b>贴界≠收敛</b>：${esc(cv.bound_vs_converge)}</li>
    </ul>
    <h4>权重口径</h4><table><tr><th>键</th><th>残差口径</th></tr>${rows}</table>
    <h4 style="margin-top:12px">元件参数 / 单位 / 默认上下界（可在参数表逐元件改写）</h4>
    <table><tr><th>元件</th><th>参数</th><th>单位</th><th>默认下界–上界</th></tr>${elrows}</table>`;
}

// ---------------- 导出/导入/清空 ----------------
async function resetDb() {
  if (!confirm("清空所有运行与候选并重建固定 fixture？")) return;
  await api("/api/reset",{method:"POST"});
  await refreshAll();
  alert("已清空并重建 fixture。");
}

async function importFile(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  try {
    const payload = JSON.parse(await file.text());
    const r = await api("/api/import",{method:"POST",body:{payload}});
    alert("导入完成：" + JSON.stringify(r.counts));
    await refreshAll(); await loadTree();
  } catch(e) { alert("导入失败：" + e.message); }
  ev.target.value = "";
}
