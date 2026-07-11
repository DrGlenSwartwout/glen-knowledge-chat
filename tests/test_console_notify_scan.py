# tests/test_console_notify_scan.py
import sqlite3

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    from dashboard import inbox as _inbox
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    sent = {}

    def _fake_bulk(to_email, subject, body, from_name=None, html=None):
        sent.update(to=to_email, subject=subject, body=body)
        return {"id": "x", "via": "ghl"}

    monkeypatch.setattr(_inbox, "send_bulk", _fake_bulk)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod, sent


def _seed(appmod):
    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx)
        cp.upsert_portal(cx, "a@x.com", "A", {})


def _auth():
    # _portal_console_ok() checks X-Console-Key header (or ?key=) against
    # CONSOLE_SECRET (see app.py:_portal_console_ok, and Task 4's
    # tests/test_console_set_current.py, which already verified this pattern).
    return {"X-Console-Key": "test-secret"}


def test_flag_off_does_not_send(client, monkeypatch):
    c, appmod, sent = client
    _seed(appmod)
    monkeypatch.delenv("PORTAL_SCAN_NOTIFY_ENABLED", raising=False)
    r = c.post("/api/console/portal/notify-scan", json={"email": "a@x.com"}, headers=_auth())
    assert r.status_code == 200
    body = r.get_json()
    assert body["sent"] is False and body["reason"] == "flag off"
    assert not sent


def test_flag_on_sends_via_bulk(client, monkeypatch):
    c, appmod, sent = client
    _seed(appmod)
    monkeypatch.setenv("PORTAL_SCAN_NOTIFY_ENABLED", "1")
    r = c.post("/api/console/portal/notify-scan", json={"email": "a@x.com"}, headers=_auth())
    assert r.status_code == 200
    body = r.get_json()
    assert body["sent"] is True
    assert sent.get("to") == "a@x.com"
    # copy rules: no em dashes, no ALL CAPS words, signed correctly
    assert "—" not in sent["body"]
    assert "In wellness,\nDr. Glen & Rae" in sent["body"]


def test_opted_out_does_not_send(client, monkeypatch):
    c, appmod, sent = client
    _seed(appmod)
    monkeypatch.setenv("PORTAL_SCAN_NOTIFY_ENABLED", "1")
    from dashboard import notify_state as ns
    with sqlite3.connect(appmod.LOG_DB) as cx:
        ns.set_opt(cx, "a@x.com", "out")
    r = c.post("/api/console/portal/notify-scan", json={"email": "a@x.com"}, headers=_auth())
    assert r.status_code == 200
    body = r.get_json()
    assert body["sent"] is False and body["reason"] == "opted out"
    assert not sent


def test_requires_auth(client):
    c, appmod, sent = client
    _seed(appmod)
    r = c.post("/api/console/portal/notify-scan", json={"email": "a@x.com"})
    assert r.status_code == 401


def test_missing_email_400(client):
    c, appmod, sent = client
    r = c.post("/api/console/portal/notify-scan", json={}, headers=_auth())
    assert r.status_code == 400
