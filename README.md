# 频弧辨证台（EIS 等效电路可辨识性对照台）

在有限观测频带内，结构不同的等效电路可以给出几乎重合的 Nyquist 弧。
本工具让使用者并排保留多个 R / C / L / CPE / Warburg 候选网络的拟合结果，
明确区分**真正的数值收敛**与**参数贴上界**，并在设计矩阵近秩亏时
**拒绝输出伪精密的标准误**。

- 后端：Python + FastAPI + NumPy（拟合器为自带有界 LM，无 scipy 依赖）
- 存储：SQLite（`data/eis.db`），可一键导出单个 JSON，清空后原样导回复核
- 前端：零依赖单页（原生 JS + canvas 绘制 Nyquist / Bode 幅相 / 加权残差）

## 安装

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## 演示 / 验收

```bash
.venv/bin/python -m pytest -q && \
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 5523
```

浏览器打开 <http://127.0.0.1:5523>，页面标题即 **频弧辨证台**。
首次启动自动播种固定 fixture 与三个候选电路。

## 数据口径（明文化）

**数据表**：CSV/TSV/空白分隔，列 `freq_hz, z_re, z_im [, sigma_re, sigma_im]`
（首行可为表头；`#` 为注释）。

- 时间约定 **e^{+jωt}**：`Z = R + jX`，**电容/CPE 虚部为负**，
  Nyquist 图画 **−Im(Z) vs Re(Z)**。
- CPE：`Z_Q = 1 / (Y0·(jω)^n)`，复幂取**主值支**
  `(jω)^n = ω^n(cos nπ/2 + j sin nπ/2)`；n=1 即电容，n=0 即电阻。
- Warburg：n=1/2 的同一主值支，`Z_W = (1−j)/(Y0·√(2ω))`，
  Y0 单位 S·s^0.5（与常见 σ 写法的换算：σ = 1/(Y0√2)）。
- CPE 指数 n 的默认边界为 **[0, 1]**，可在参数表里逐元件改写。
- **频率 f ≤ 0（或 NaN/解析失败）的行不进入对数轴与拟合**，
  但在“数据口径”页与每次运行详情里报出其**原始物理行号**与数据序号。
- 权重三选一：`sigma`（数据列标准差）、`unit`（等权）、`modulus`（σ=|Zobs|）。
  残差向量按 Re/Im 拼接：`r=(Zre−zre)/σre, (Zim−zim)/σim`。

**贴界 ≠ 收敛**：

- `converged`：logit 无界空间归一化梯度与步长同时满足阈值，且无自由参数贴边；
- `bound_only`：有自由参数位于边界邻域（到边界距离 ≤ 1e-6·尺度），
  即便残差不再下降也**不宣称收敛**——优化想继续但被边界挡住；
- `max_iter / stalled / failed` 分别表示迭代用尽、信任域停滞、数值失败。

**秩亏不给伪精密 SE**：设计矩阵 `J = dr/dx` 先按各参数上下界区间标定，
再做 SVD。当最小奇异值 `< 1e-8 × 最大奇异值` 时判为数值秩缺，
此时所有标准误返回 `null`，只给出参数相关方向与**零空间向量**
（例如双串联电阻的零方向 `[0.707, −0.707, 0, 0]` 表示“只有电阻之和可辨识”）。

## 稳定电路表达

`+` 串联、`|` 并联（`|` 优先于 `+`，括号显式分组）。
后端在**结合律展平后对同级子网络按字典序排序**，得到与书写顺序无关的
稳定表达（canonical），用于识别同一拓扑，例如
`Rs+(Rct|Cdl)` 与 `(Cdl|Rct)+Rs` 的稳定表达相同。

参数带单位、物理上下界、固定勾选与共享组（同名共享组合并为一个自由变量；
跨元件种类或界区间无交集会被拒绝）。

## 固定 fixture 与验收点

`fixtures/randles_rc.csv` 由固定种子（`eis/fixtures.py` 中 `SEED`）生成，
逐字节可重放：真值 `Rs=10Ω, Rct=100Ω, Cdl=100µF`，1 Hz–100 kHz 共 41 点，
约 0.5% 噪声，另插入 **f=0（物理行 10）** 与 **f=−12（物理行 28）** 两条脏行。

三个候选（`③ 拟合对照`页一键并排拟合）：

1. `C1  Rs+(Rct|Cdl)`：同构 RC，**真正收敛**，满秩并给出有限 SE；
2. `C2  Rs+(Rct|Q1)`：结构不同的 R-CPE，频带内 χ² 与 C1 几乎相同，
   但 **Q1.n 贴着用户上界 0.999**，状态为 `bound_only`；
3. `C3  Rs+Rleak+(Rct|Cdl)`：两个串联电阻只有和可辨识，
   **数值秩 3/4、条件数巨大**，SE 全部为 `null` 并给出零空间方向。

多起点（默认 12，确定性种子 + 数据启发专家点 + 拉丁超立方）结果在卡片上
**并排保留**，蓝框标出最优 χ²；点“完整详情”查看每个起点的 χ²、状态、贴界。

## 版本（父子）与可追溯子版本

- “固定参数重跑（子版本）”：勾选参数并给固定值，父版本原样保留，
  新版本挂在父节点下，`fixed_note` 记录固定内容。
- “改权重生成子版本”：切换 sigma/unit/modulus 重跑，形成可追溯子版本。
- `④ 版本树`页查看父子链；每次运行的曲线、残差、多起点全部持久化。

## 运行记录导出 / 清空 / 重放复核

- 右上角 **导出运行记录(JSON)**（或 `GET /api/export/download`）
  导出全库快照（保留所有主键与父子引用）。
- **清空并重建 fixture**（或 `POST /api/reset`）回到播种状态。
- 用 **导入复核**（或 `POST /api/import`，body `{"payload": <导出JSON>}`）
  清空后原样写回；`tests/test_replay.py` 自动断言 χ²、状态、秩标记、
  参数值与父子关系在导出→清空→导入后完全一致。

## 目录

```
app.py                 FastAPI 入口与路由
eis/circuit.py         文法解析 / 稳定表达 / 元件目录与默认界
eis/impedance.py       元件阻抗（CPE 主值支、虚部符号约定）
eis/fitting.py         有界 LM、多起点、SVD 秩/SE/相关/零空间
eis/dataio.py          数据导入与 f≤0 行号报告
eis/db.py              SQLite、播种、导出/导入/清空
eis/service.py         拟合编排与视图数据组装
eis/fixtures.py        固定 fixture 与候选（固定种子）
fixtures/randles_rc.csv
static/                单页前端（index.html / app.js / app.css）
tests/                 33 个自动化测试
```

## API 摘要

`GET /api/health` · `GET /api/catalog`（全部数值约定）·
数据集 `/api/datasets` · 候选 `/api/circuits`（`POST` 预览/新建，`PATCH` 改参数）·
`POST /api/fit` · `POST /api/fit/child`（fix / reweight）·
`GET /api/runs/{id}`（含曲线、残差、多起点、相关、零空间）·
`GET /api/runs/tree` · `GET /api/export` · `POST /api/import` · `POST /api/reset`。
