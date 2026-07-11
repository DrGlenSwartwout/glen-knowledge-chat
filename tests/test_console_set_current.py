# tests/test_console_set_current.py
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


def _seed(appmod):
    from dashboard import client_portal as cp, portal_biofield_reports as pbr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); pbr.init_table(cx)
        cp.upsert_portal(cx, "a@x.com", "A", {})
        pbr.upsert_report(cx, "a@x.com", "2026-07-02", "1", {}, "confirmed")


def _auth():
    # _portal_console_ok() checks X-Console-Key header (or ?key= query param)
    # against CONSOLE_SECRET — verified against app.py's _portal_console_ok and
    # tests/test_console_biofield_portal.py's fixture, which sets the same secret.
    return {"X-Console-Key": "test-secret"}


def test_set_current_ok(client):
    c, appmod = client; _seed(appmod)
    r = c.post("/api/console/portal/set-current",
               json={"email": "a@x.com", "scan_date": "2026-07-02"}, headers=_auth())
    assert r.status_code == 200
    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert cp.get_current_scan(cx, "a@x.com") == "2026-07-02"


def test_set_current_unknown_date_400(client):
    c, appmod = client; _seed(appmod)
    r = c.post("/api/console/portal/set-current",
               json={"email": "a@x.com", "scan_date": "2030-01-01"}, headers=_auth())
    assert r.status_code == 400


def test_set_current_missing_fields_400(client):
    c, appmod = client; _seed(appmod)
    r = c.post("/api/console/portal/set-current", json={"email": "a@x.com"}, headers=_auth())
    assert r.status_code == 400


def test_set_current_no_portal_404(client):
    c, appmod = client
    from dashboard import portal_biofield_reports as pbr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        pbr.init_table(cx)
        pbr.upsert_report(cx, "nobody@x.com", "2026-07-02", "1", {}, "confirmed")
    r = c.post("/api/console/portal/set-current",
               json={"email": "nobody@x.com", "scan_date": "2026-07-02"}, headers=_auth())
    assert r.status_code == 404


def test_set_current_requires_auth(client):
    c, appmod = client; _seed(appmod)
    r = c.post("/api/console/portal/set-current",
               json={"email": "a@x.com", "scan_date": "2026-07-02"})
    assert r.status_code == 401
