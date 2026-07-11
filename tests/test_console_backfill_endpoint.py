# tests/test_console_backfill_endpoint.py
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


def _auth():
    # _portal_console_ok() checks X-Console-Key header (or ?key= query param)
    # against CONSOLE_SECRET — mirrors tests/test_console_set_current.py.
    return {"X-Console-Key": "test-secret"}


def _seed_two_portals(appmod):
    from dashboard import client_portal as cp, portal_biofield_reports as pbr
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cp.init_client_portal_table(cx); pbr.init_table(cx)
        # fresh portal: no auto_advance / current_scan_date set yet
        cp.upsert_portal(cx, "a@x.com", "A", {})
        # pinned + opted-out portal: must be left alone
        cp.upsert_portal(cx, "b@x.com", "B",
                          {"auto_advance": False, "current_scan_date": "2026-07-02"})
        for email in ("a@x.com", "b@x.com"):
            pbr.upsert_report(cx, email, "2026-07-02", "1", {}, "confirmed")
            pbr.upsert_report(cx, email, "2026-07-09", "2", {}, "confirmed")


def test_backfill_endpoint_requires_auth(client):
    c, appmod = client
    _seed_two_portals(appmod)
    r = c.post("/api/console/portal/backfill-scan-history")
    assert r.status_code == 401


def test_backfill_endpoint_updates_fresh_portal_only(client):
    c, appmod = client
    _seed_two_portals(appmod)
    r = c.post("/api/console/portal/backfill-scan-history", headers=_auth())
    assert r.status_code == 200
    body = r.get_json()
    assert body == {"ok": True, "portals": 2, "updated": 1}

    from dashboard import client_portal as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert cp.get_current_scan(cx, "a@x.com") == "2026-07-09"
        assert cp.get_auto_advance(cx, "a@x.com") is True
        # pinned/opted-out portal is untouched
        assert cp.get_current_scan(cx, "b@x.com") == "2026-07-02"
        assert cp.get_auto_advance(cx, "b@x.com") is False


def test_backfill_endpoint_idempotent(client):
    c, appmod = client
    _seed_two_portals(appmod)
    r1 = c.post("/api/console/portal/backfill-scan-history", headers=_auth())
    assert r1.status_code == 200
    assert r1.get_json()["updated"] == 1

    r2 = c.post("/api/console/portal/backfill-scan-history", headers=_auth())
    assert r2.status_code == 200
    body2 = r2.get_json()
    assert body2 == {"ok": True, "portals": 2, "updated": 0}
