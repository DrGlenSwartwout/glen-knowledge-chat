import os, pytest
if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("needs doppler env for import app", allow_module_level=True)
import app as appmod

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    monkeypatch.setattr(appmod, "GLEN_CONSULT_HOURS", "1-7:09:00-17:00", raising=False)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()

ADMIN = {"X-Console-Key": "test-secret"}

def test_consult_ready_flip_requires_auth(client):
    r = client.post("/api/console/consult-ready", json={"email": "c@x.com", "ready": True})
    assert r.status_code == 401

def test_consult_ready_flip_sets_flag(client):
    r = client.post("/api/console/consult-ready",
                    json={"email": "c@x.com", "ready": True}, headers=ADMIN)
    assert r.status_code == 200 and r.get_json()["ready"] is True
    import sqlite3
    from dashboard import consult
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert consult.consult_is_ready(cx, "c@x.com") is True
