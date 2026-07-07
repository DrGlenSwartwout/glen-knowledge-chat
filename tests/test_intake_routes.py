import os
os.environ.setdefault("DATA_DIR", "/tmp/intake-rt")
import json, importlib, sqlite3, pytest


@pytest.fixture
def client(monkeypatch):
    import app as appmod
    importlib.reload(appmod)
    from dashboard import intake

    class _Ident:  # stand-in for portal_identity.Identity
        def __init__(self, email): self.email = email

    # any token "good" resolves to a fixed member; "" or "bad" -> None
    def fake_ident(cx, token):
        return _Ident("member@x.com") if token == "good" else None
    monkeypatch.setattr(appmod, "_evox_ident", fake_ident)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_form_endpoint_returns_sections(client):
    r = client.get("/api/intake/form")
    assert r.status_code == 200
    assert r.get_json()["version"]
    assert any(s["id"] == "dimensions" for s in r.get_json()["sections"])


def test_state_bad_token_404(client):
    r = client.get("/api/intake/state?token=bad")
    assert r.status_code == 404 and r.get_json()["error"] == "not_found"


def test_token_gate_precedes_body_on_submit(client):
    r = client.post("/api/intake/submit?token=bad", json={"garbage": True})
    assert r.status_code == 404  # token wins over validation


def test_save_draft_then_state(client):
    client.post("/api/intake/save-draft?token=good", json={"answers": {"first_name": "Ann"}})
    r = client.get("/api/intake/state?token=good")
    body = r.get_json()
    assert body["status"] == "draft" and body["submitted"] is False
    assert body["answers"]["first_name"] == "Ann"


def test_submit_validation_error_lists_fields(client):
    r = client.post("/api/intake/submit?token=good", json={"answers": {}})
    assert r.status_code == 400
    assert "first_name" in r.get_json()["errors"]


def test_submit_success_then_double_submit_409(client):
    good = {"answers": {
        "first_name": "Ann", "last_name": "Lee", "email": "a@x.com", "dob": "1970-01-01",
        "terrain": 1, "penetration": 5, "tissue_layer": 3, "response": 3, "commitment": 8,
        "terms": {"agreed": True, "signature": "Ann Lee", "date": "2026-07-07"}}}
    assert client.post("/api/intake/submit?token=good", json=good).status_code == 200
    assert client.get("/api/intake/state?token=good").get_json()["submitted"] is True
    r2 = client.post("/api/intake/submit?token=good", json=good)
    assert r2.status_code == 409 and r2.get_json()["error"] == "already_submitted"
