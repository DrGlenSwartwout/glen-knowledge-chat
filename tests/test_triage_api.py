import os, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod, "send_evox_email", lambda *a, **k: ("console-log", None), raising=False)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

ADMIN = {"X-Console-Key": "test-secret"}

def test_triage_invite_requires_auth(client):
    r = client.post("/api/console/triage-invite", json={"email": "p@x.com", "name": "P", "practitioner": "glen"})
    assert r.status_code == 401

def test_triage_invite_bad_practitioner(client):
    r = client.post("/api/console/triage-invite",
                    json={"email": "p@x.com", "name": "P", "practitioner": "bob"}, headers=ADMIN)
    assert r.status_code == 400

def test_triage_invite_creates_and_returns_url(client):
    r = client.post("/api/console/triage-invite",
                    json={"email": "p@x.com", "name": "Pat", "practitioner": "rae"}, headers=ADMIN)
    assert r.status_code == 200
    d = r.get_json()
    assert d["ok"] is True and "/triage/" in d["url"]
