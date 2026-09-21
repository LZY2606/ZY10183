"""导出 -> 清空 -> 重新导入 -> 复核 的闭环。"""
import pytest


def _fit_three(client, ds, n=6):
    cids = [c["id"] for c in client.get("/api/circuits").json()]
    ids = []
    for cid in cids:
        r = client.post("/api/fit", json={
            "circuit_id": cid, "dataset_id": ds,
            "weighting": "sigma", "n_starts": n}).json()
        ids.append(r["id"])
    return ids


def test_export_reset_import_roundtrip(client):
    ds = client.get("/api/datasets").json()[0]["id"]
    ids = _fit_three(client, ds)
    # 加一个子版本，验证父子引用也能跨导出保留
    child = client.post("/api/fit/child", json={
        "parent_id": ids[0], "action": "reweight", "weighting": "modulus",
        "n_starts": 4,
        "circuit_id": client.get("/api/circuits").json()[0]["id"],
        "dataset_id": ds}).json()["id"]

    payload = client.get("/api/export").json()
    before = {rid: client.get(f"/api/runs/{rid}").json()["run"]
              for rid in ids + [child]}

    # 清空（回到仅 fixture）
    client.post("/api/reset")
    assert client.get(f"/api/runs/{ids[0]}").status_code == 404

    # 原样导回
    r = client.post("/api/import", json={"payload": payload})
    assert r.status_code == 200, r.text
    assert r.json()["counts"]["runs"] >= 4

    # 复核：chi2 / 状态 / 父子关系 / 参数值全部一致
    for rid, run_before in before.items():
        run_after = client.get(f"/api/runs/{rid}").json()["run"]
        assert run_after["chi2"] == pytest.approx(run_before["chi2"])
        assert run_after["status"] == run_before["status"]
        assert run_after["near_singular"] == run_before["near_singular"]
        assert run_after["parent_id"] == run_before["parent_id"]
        pa = {p["name"]: p["value"] for p in run_after["params"]}
        pb = {p["name"]: p["value"] for p in run_before["params"]}
        assert pa == pytest.approx(pb)
    tree = client.get("/api/runs/tree").json()
    assert child in tree["children"][ids[0]]


def test_import_rejects_bad_format(client):
    r = client.post("/api/import", json={"payload": {"format": "nope", "tables": {}}})
    assert r.status_code == 400


def test_fixture_file_is_fixed_and_reproducible():
    from eis.fixtures import generate_fixture_text
    a = generate_fixture_text()
    b = generate_fixture_text()
    assert a == b  # 固定种子，逐字节可重放
    assert "\n0,100,-50" in a and "\n-12,100,-50" in a
