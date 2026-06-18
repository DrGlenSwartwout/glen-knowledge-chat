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
