import numpy as np


def test_title_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "频弧辨证台" in r.text


def test_seeded_fixtures(client, seeded_ids):
    ds = client.get(f"/api/datasets/{seeded_ids['dataset']}").json()
    assert len(ds["freqs"]) == 25
    excluded = {e["line"] for e in ds["excluded"]}
    assert excluded == {2, 3}  # CSV 首行表头，零频第 2 行、负频第 3 行
    assert len(client.get("/api/circuits").json()) == 3


def test_expr_validation_api(client):
    r = client.post("/api/validate/expr", json={"expr": "R, P(R:CT, CPE:DL)"})
    assert r.status_code == 200
    assert r.json()["canonical"] == "s(r,p(r:ct,cpe:dl))"
    bad = client.post("/api/validate/expr", json={"expr": "r,r"})
    assert bad.status_code == 400


def test_full_fit_refit_parent_child_flow(client, seeded_ids):
    ds, cid_b = seeded_ids["dataset"], seeded_ids["B"]
    r = client.post("/api/fit", json={
        "dataset_id": ds, "circuit_id": cid_b,
        "weight_mode": "modulus", "gamma": 0, "n_starts": 8,
        "max_iter": 300, "seed": 1186}).json()
    parent_id = r["id"]
    res = r["result"]
    assert res["reason"] == "at_bound"
    assert "n:dl" in res["bound_states"][0]["key"] or any(
        b["key"] == "n:dl" and b["at_upper"] for b in res["bound_states"])

    # 固定 n:dl 重跑 -> 子版本，parent_id 指向父运行
    nval = res["values"]["n:dl"]
    r2 = client.post("/api/refit", json={
        "parent_id": parent_id, "fixed": {"n:dl": nval},
        "n_starts": 4, "label": "固定 n"}).json()
    assert r2["id"] != parent_id
    detail = client.get(f"/api/runs/{r2['id']}").json()
    assert detail["parent_id"] == parent_id
    assert detail["result"]["fixed"] == {"n:dl": nval}
    # 父版本仍在、未被覆盖
    parent_detail = client.get(f"/api/runs/{parent_id}").json()
    assert parent_detail["id"] == parent_id
    assert "fixed" not in parent_detail["result"]


def test_gamma_child_version(client, seeded_ids):
    ds, cid = seeded_ids["dataset"], seeded_ids["A"]
    p = client.post("/api/fit", json={
        "dataset_id": ds, "circuit_id": cid, "gamma": 0,
        "n_starts": 6, "max_iter": 200}).json()
    ch = client.post("/api/refit", json={
        "parent_id": p["id"], "fixed": {}, "gamma": 0.5,
        "label": "γ=0.5", "n_starts": 6}).json()
    child = client.get(f"/api/runs/{ch['id']}").json()
    assert child["parent_id"] == p["id"] and child["gamma"] == 0.5


def test_export_import_roundtrip_and_wipe(client, seeded_ids):
    fit = client.post("/api/fit", json={
        "dataset_id": seeded_ids["dataset"], "circuit_id": seeded_ids["A"],
        "n_starts": 4, "max_iter": 100}).json()
    bundle = client.get("/api/export").json()
    runs_before = client.get("/api/runs").json()
    assert any(r["id"] == fit["id"] for r in runs_before)

    # 导出包保留原 ID；清空库后再导入，运行记录逐 ID 复核
    client.post("/api/reset")
    assert client.get("/api/runs").json() == []
    counts = client.post("/api/import", json=bundle).json()["imported"]
    assert counts["runs"] >= 1
    restored = client.get(f"/api/runs/{fit['id']}").json()
    assert restored["result"]["sse"] == fit["result"]["sse"]
    assert restored["result"]["expr_canonical"] == fit["result"]["expr_canonical"]
    # 再重置恢复干净 fixture，保证测试可重复
    client.post("/api/reset")


def test_dataset_upload_with_bad_rows(client):
    csv = "freq_hz,zreal,zimag\n-1,1,1\n0,2,2\n1,50,-1\n10,40,-5"
    r = client.post("/api/datasets", json={"name": "t", "csv_text": csv}).json()
    assert len(r["excluded"]) == 2
    ds = client.get(f"/api/datasets/{r['id']}").json()
    assert len(ds["freqs"]) == 2
