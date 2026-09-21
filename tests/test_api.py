def test_index_title(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "频弧辨证台" in r.text


def test_health_and_catalog(client):
    assert client.get("/api/health").json()["title"] == "频弧辨证台"
    cat = client.get("/api/catalog").json()
    assert {"R", "C", "L", "Q", "W"} <= set(cat["elements"])
    assert "主值支" in cat["conventions"]["cpe"]


def test_seeded_fixture_and_excluded_rows(client):
    ds = client.get("/api/datasets").json()[0]
    assert ds["n_ok"] == 41 and ds["n_excluded"] == 2
    detail = client.get(f"/api/datasets/{ds['id']}").json()
    bad = [p for p in detail["points"] if p["status"] != "ok"]
    assert {p["line_no"] for p in bad} == {10, 28}
    assert all(p["freq"] <= 0 for p in bad)


def test_circuit_crud_and_patch(client):
    r = client.post("/api/circuits", json={
        "name": "tmp", "text": "Rs+(Rct|W1)"})
    assert r.status_code == 200, r.text
    cid = r.json()["id"]
    body = client.get(f"/api/circuits/{cid}").json()
    assert body["canonical"] == "Rs+Rct|W1"
    # 参数上下界非法 -> 400
    for p in body["parameters"]:
        if p["name"] == "Rs.R":
            p["lower"], p["upper"] = 10, 1
    bad = client.patch(f"/api/circuits/{cid}", json={"parameters": body["parameters"]})
    assert bad.status_code == 400


def test_fit_lifecycle_bound_flag_and_child_versions(client):
    ds = client.get("/api/datasets").json()[0]["id"]
    cids = {c["name"]: c["id"] for c in client.get("/api/circuits").json()}
    # 三个候选各拟合一次（少起点保证测试快）
    runs = {}
    for key in cids:
        r = client.post("/api/fit", json={
            "circuit_id": cids[key], "dataset_id": ds,
            "weighting": "sigma", "n_starts": 6})
        assert r.status_code == 200, r.text
        runs[key] = r.json()
    c2 = next(v for k, v in runs.items() if k.startswith("C2"))
    assert c2["status"] == "bound_only"
    assert "Q1.n" in c2["at_upper"]
    c3 = next(v for k, v in runs.items() if k.startswith("C3"))
    assert c3["near_singular"] is True
    c1 = next(v for k, v in runs.items() if k.startswith("C1"))
    assert c1["status"] == "converged"
    # 详情：满秩有 SE，秩缺无 SE
    d1 = client.get(f"/api/runs/{c1['id']}").json()
    assert all(p["se"] is not None for p in d1["run"]["params"])
    assert len(d1["run"]["starts"]) == 6
    d3 = client.get(f"/api/runs/{c3['id']}").json()
    assert all(p["se"] is None for p in d3["run"]["params"])
    assert d3["run"]["null_directions"]

    # 固定参数子版本
    child = client.post("/api/fit/child", json={
        "parent_id": c1["id"], "action": "fix",
        "fixed": {"Rs.R": 10.0}, "weighting": "sigma", "n_starts": 4,
        "circuit_id": cids[next(k for k in cids if k.startswith("C1"))],
        "dataset_id": ds}).json()
    assert child["parent_id"] == c1["id"]
    detail = client.get(f"/api/runs/{child['id']}").json()["run"]
    assert detail["fixed_note"].startswith("固定")
    # 父版本仍在
    assert client.get(f"/api/runs/{c1['id']}").status_code == 200

    # 改权重子版本
    rw = client.post("/api/fit/child", json={
        "parent_id": c1["id"], "action": "reweight",
        "weighting": "modulus", "n_starts": 4,
        "circuit_id": cids[next(k for k in cids if k.startswith("C1"))],
        "dataset_id": ds}).json()
    assert rw["parent_id"] == c1["id"]
    assert client.get(f"/api/runs/{rw['id']}").json()["run"]["weighting"] == "modulus"

    tree = client.get("/api/runs/tree").json()
    child_ids = set(tree["children"].get(c1["id"], []))
    assert {child["id"], rw["id"]} <= child_ids


def test_dataset_upload_rejects_all_invalid(client):
    r = client.post("/api/datasets", json={
        "name": "bad", "text": "f,re,im\n0,1,1\n-2,3,4"})
    assert r.status_code == 400


def test_equivalent_circuits_near_identical_chi2(client):
    ds = client.get("/api/datasets").json()[0]["id"]
    cids = {c["name"]: c["id"] for c in client.get("/api/circuits").json()}
    chi = []
    for prefix in ("C1", "C2"):
        cid = cids[next(k for k in cids if k.startswith(prefix))]
        r = client.post("/api/fit", json={
            "circuit_id": cid, "dataset_id": ds,
            "weighting": "sigma", "n_starts": 8}).json()
        chi.append(r["chi2"])
    assert abs(chi[0] - chi[1]) / chi[0] < 0.05
