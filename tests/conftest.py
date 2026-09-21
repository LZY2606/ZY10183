import os
import tempfile
import pytest

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["PINPAI_DB"] = _tmp.name

from pinpai.main import app, conn  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def seeded_ids(client):
    ds = client.get("/api/datasets").json()[0]
    circs = client.get("/api/circuits").json()
    return {"dataset": ds["id"], "A": circs[0]["id"],
            "B": circs[1]["id"], "C": circs[2]["id"]}
