"""阻抗表数据导入口径。

接受 CSV/TSV/空白分隔文本，列顺序为：

    freq_hz, z_re, z_im [, sigma_re, sigma_im]

* 第一行若含非数字字段则视为表头（列名大小写不敏感，``freq/f/Hz``、
  ``zre/re/real``、``zim/im/imag``、``sigma/err/w`` 均可识别）。
* ``#`` 开头为注释，空行忽略。
* 频率 **<= 0** 的行不进入对数轴与拟合，但会在结果中报出其
  **原始行号**（文件中的物理行号，1 起始，含表头/注释行计数）。
* 有效行按频率升序排列，原始顺序同时保留（``rownum`` 为去除表头后的
  数据行序号）。
* sigma 为测量标准差（欧姆），用于 sigma 加权；缺省时 sigma_re=sigma_im=1。
"""

from __future__ import annotations

import math
import re

DATA_HEADERS = ("freq_hz", "z_re", "z_im", "sigma_re", "sigma_im")
_SPLIT = re.compile(r"[,\t;]+")


def _split_fields(line: str) -> list[str]:
    if _SPLIT.search(line):
        return [x.strip() for x in _SPLIT.split(line.strip()) if x.strip() != ""]
    return line.split()


def _is_header(fields: list[str]) -> bool:
    for f in fields:
        try:
            float(f)
        except ValueError:
            return True
    return False


def parse_table(text: str) -> dict:
    lines = text.replace("\ufeff", "").splitlines()
    header: list[str] | None = None
    rows: list[dict] = []
    data_ordinal = 0

    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = _split_fields(line)
        if not fields:
            continue
        if header is None and _is_header(fields):
            header = fields
            continue
        data_ordinal += 1
        row = {"line": line_no, "rownum": data_ordinal}
        try:
            if len(fields) < 3:
                raise ValueError(f"至少需要 3 列(freq,zre,zim)，实际 {len(fields)} 列")
            nums = [float(x) for x in fields]
        except ValueError as exc:
            row.update(status="excluded", reason=f"无法解析为数值: {exc}",
                       freq=None, z_re=None, z_im=None,
                       sigma_re=None, sigma_im=None)
            rows.append(row)
            continue
        freq, zre, zim = nums[0], nums[1], nums[2]
        sre = sim = 1.0
        if len(nums) >= 4:
            sre = nums[3]
            sim = nums[4] if len(nums) >= 5 else nums[3]
        row.update(freq=freq, z_re=zre, z_im=zim, sigma_re=sre, sigma_im=sim)
        if not math.isfinite(freq) or not math.isfinite(zre) or not math.isfinite(zim):
            row.update(status="excluded", reason="含 NaN/Inf")
        elif freq <= 0:
            row.update(status="excluded", reason="频率非正(f<=0)，不进入对数轴")
        elif sre is not None and (not math.isfinite(sre) or sre <= 0
                                  or not math.isfinite(sim) or sim <= 0):
            row.update(status="excluded", reason="sigma 必须为正数")
        else:
            row.update(status="ok", reason="")
        rows.append(row)

    valid = sorted(
        (r for r in rows if r["status"] == "ok"),
        key=lambda r: r["freq"],
    )
    return {
        "header": header,
        "rows": rows,
        "valid": valid,
        "valid_count": len(valid),
        "excluded": [r for r in rows if r["status"] == "excluded"],
        "excluded_count": sum(1 for r in rows if r["status"] == "excluded"),
    }
