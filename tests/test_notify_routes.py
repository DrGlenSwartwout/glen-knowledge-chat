# tests/test_notify_routes.py
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


def test_portal_notify_pref_sets_opt(client):
    c, appmod = client
    tok = _seed_portal(appmod, "p@y.com", "P", {"layers": []})
    assert c.post(f"/api/portal/{tok}/notify-pref", json={"pref": "out"}).status_code == 200
    import sqlite3
    from dashboard import notify_state as N
    assert N.get_state(sqlite3.connect(appmod.LOG_DB), "p@y.com")["opt_status"] == "out"


def test_unsubscribe_link_opts_out(client):
    c, appmod = client
    tok = _seed_portal(appmod, "u@y.com", "U", {"layers": []})
    assert c.get(f"/unsubscribe?token={tok}").status_code == 200
    import sqlite3
    from dashboard import notify_state as N
    assert N.get_state(sqlite3.connect(appmod.LOG_DB), "u@y.com")["opt_status"] == "out"


def test_twilio_inbound_stop_start(client):
    c, appmod = client
    import sqlite3
    from dashboard import notify_state as N
    cx = sqlite3.connect(appmod.LOG_DB); N.set_phone(cx, "t@y.com", "+15551230000"); cx.commit()
    c.post("/sms/inbound", data={"From": "+15551230000", "Body": "STOP"})
    assert N.get_state(sqlite3.connect(appmod.LOG_DB), "t@y.com")["opt_status"] == "out"
    c.post("/sms/inbound", data={"From": "+15551230000", "Body": "START"})
    assert N.get_state(sqlite3.connect(appmod.LOG_DB), "t@y.com")["opt_status"] == "in"


def test_open_marks_engaged(client):
    c, appmod = client
    tok = _seed_portal(appmod, "e@y.com", "E", {"layers": [{"n": 1, "title": "t"}]})
    c.get(f"/api/portal/{tok}")
    import sqlite3
    from dashboard import notify_state as N
    assert N.get_state(sqlite3.connect(appmod.LOG_DB), "e@y.com")["engaged"] is True


def test_admin_notify_state_returns_decision_and_link(client):
    c, appmod = client
    j = c.post("/api/admin/notify-state?key=test-secret",
               json={"email": "ns@y.com", "name": "NS"}).get_json()
    assert j["eligible"] is True and j["variant"] == 0
    assert j["url"].startswith("https://") and "/portal/" in j["url"]
    assert "token=" in j["unsubscribe"]
    url1 = j["url"]
    j2 = c.post("/api/admin/notify-state?key=test-secret", json={"email": "ns@y.com"}).get_json()
    assert j2["url"] == url1                                   # stable: same link, not rotated
    c.post("/api/admin/notify-sent?key=test-secret", json={"email": "ns@y.com"})
    j3 = c.post("/api/admin/notify-state?key=test-secret", json={"email": "ns@y.com"}).get_json()
    assert j3["variant"] == 1                                  # taper advanced


def test_admin_notify_state_requires_key(client):
    c, _ = client
    assert c.post("/api/admin/notify-state", json={"email": "x@y.com"}).status_code == 401


def test_ensure_token_link_actually_loads(client):
    c, appmod = client
    j = c.post("/api/admin/notify-state?key=test-secret",
               json={"email": "ld@y.com", "name": "LD"}).get_json()
    tok = j["url"].rstrip("/").split("/")[-1]
    assert c.get(f"/api/portal/{tok}").status_code == 200      # the minted token resolves


def test_sms_status_records_delivery(client):
    c, appmod = client
    c.post("/sms/status", data={"MessageSid": "SMx", "MessageStatus": "delivered",
                                "To": "+18085550000", "ErrorCode": ""})
    j = c.get("/api/admin/sms-deliveries?key=test-secret").get_json()
    assert any(d["message_sid"] == "SMx" and d["status"] == "delivered" for d in j["deliveries"])


def test_sms_deliveries_failed_filter_and_auth(client):
    c, _ = client
    c.post("/sms/status", data={"MessageSid": "SMok", "MessageStatus": "delivered", "To": "+1"})
    c.post("/sms/status", data={"MessageSid": "SMbad", "MessageStatus": "failed", "To": "+2", "ErrorCode": "30007"})
    j = c.get("/api/admin/sms-deliveries?key=test-secret&failed=1").get_json()
    sids = [d["message_sid"] for d in j["deliveries"]]
    assert "SMbad" in sids and "SMok" not in sids
    assert c.get("/api/admin/sms-deliveries").status_code == 401      # gated
