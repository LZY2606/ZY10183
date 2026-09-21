import os
import tempfile

import numpy as np
import pytest

# 在导入 app 前指定独立数据库文件。
_tmp = tempfile.mkdtemp(prefix="eis_test_")
os.environ["EIS_DB"] = os.path.join(_tmp, "test.db")

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
from eis.db import connect, seed_if_empty  # noqa: E402


@pytest.fixture()
def conn():
    c = connect(":memory:")
    seed_if_empty(c, force=True)
    return c


@pytest.fixture()
def client():
    seed_if_empty(app_module.db(), force=True)
    return TestClient(app_module.app)


@pytest.fixture()
def arrays():
    from eis.dataio import parse_table
    from eis.fixtures import FIXTURE_PATH
    tab = parse_table(open(FIXTURE_PATH).read())
    v = tab["valid"]
    f = np.array([r["freq"] for r in v])
    z = np.array([complex(r["z_re"], r["z_im"]) for r in v])
    sr = np.array([r["sigma_re"] for r in v])
    si = np.array([r["sigma_im"] for r in v])
    return tab, f, z, sr, si
