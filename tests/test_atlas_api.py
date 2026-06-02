import json
import importlib
import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    import atlas_store
    monkeypatch.setattr(atlas_store, "CONCEPTS_PATH", tmp_path / "atlas-concepts.json")
    monkeypatch.setattr(atlas_store, "PENDING_PATH", tmp_path / "atlas-pending.json")
    (tmp_path / "atlas-concepts.json").write_text(json.dumps({"version": "t", "concepts": [
        {"id": "biofield", "label": "Biofield", "aliases": ["energy field"],
         "summary": "organizing field", "namespaces": [], "cluster": "em",
         "parent": "em", "coords": {"x": 0.4, "y": 0.5}, "neighbors": [],
         "links": [], "status": "live"}]}))
    (tmp_path / "atlas-pending.json").write_text(json.dumps({"version": "t", "concepts": [
        {"id": "detox", "label": "Detox", "aliases": [], "summary": "elimination",
         "namespaces": [], "cluster": "f", "parent": "f", "coords": {"x": 0.1, "y": 0.2},
         "neighbors": [], "links": [], "status": "pending"}]}))
    import app as app_module
    import dashboard
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "testsecret", raising=False)
    monkeypatch.setattr(dashboard, "CONSOLE_SECRET", "testsecret", raising=False)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_atlas_data_returns_live_graph(client):
    r = client.get("/atlas/data")
    assert r.status_code == 200
    body = r.get_json()
    assert [c["id"] for c in body["concepts"]] == ["biofield"]
    assert "em" in body["hierarchy"]


def test_atlas_ask_returns_concept_ids(client):
    r = client.post("/atlas/ask", json={"question": "tell me about the energy field"})
    assert r.status_code == 200
    body = r.get_json()
    assert "biofield" in body["concept_ids"]
    assert isinstance(body["answer"], str)


def test_admin_approve_requires_key(client):
    r = client.post("/admin/atlas/approve", json={"id": "detox"})
    assert r.status_code in (401, 403)


def test_admin_approve_with_key_moves_to_live(client):
    r = client.post("/admin/atlas/approve?key=testsecret", json={"id": "detox"})
    assert r.status_code == 200
    g = client.get("/atlas/data").get_json()
    assert "detox" in [c["id"] for c in g["concepts"]]
