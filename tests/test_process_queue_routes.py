# tests/test_process_queue_routes.py
import sqlite3
import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _seed_portal(appmod, email="brooke@example.com", name="Brooke Webb", content=None):
    from dashboard import client_portal as cp
    content = content or {"layers": []}
    cx = sqlite3.connect(appmod.LOG_DB)
    cp.init_client_portal_table(cx)
    token, _ = cp.upsert_portal(cx, email, name, content)
    cx.close()
    return token


def test_portal_process_request_enqueues(client):
    c, appmod = client
    tok = _seed_portal(appmod, "pr@y.com", "PR", {"biofield_status": "pending"})
    assert c.post(f"/api/portal/{tok}/process-request").status_code == 200
    j = c.get("/api/admin/process-requests?key=test-secret").get_json()
    assert any(x["email"] == "pr@y.com" for x in j["pending"])


def test_admin_mark_done(client):
    c, appmod = client
    tok = _seed_portal(appmod, "pd@y.com", "PD", {"biofield_status": "pending"})
    c.post(f"/api/portal/{tok}/process-request")
    assert c.post("/api/admin/process-request/done?key=test-secret",
                  json={"email": "pd@y.com"}).status_code == 200
    j = c.get("/api/admin/process-requests?key=test-secret").get_json()
    assert all(x["email"] != "pd@y.com" for x in j["pending"])


def test_admin_process_requests_requires_key(client):
    c, _ = client
    assert c.get("/api/admin/process-requests").status_code == 401


def test_notify_state_exposes_engaged(client):
    c, appmod = client
    import sqlite3
    from dashboard import notify_state as N
    cx = sqlite3.connect(appmod.LOG_DB); N.mark_engaged(cx, "eng@y.com"); cx.commit()
    j = c.post("/api/admin/notify-state?key=test-secret", json={"email": "eng@y.com"}).get_json()
    assert j["engaged"] is True
