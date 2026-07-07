import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


_LAYER = {"n": 1, "title": "Calm", "meaning": "settle", "remedy": "Terrain Restore",
          "dosing": "10 drops 3x/day"}
_F = [{"code": "ED3", "name": "Cell Driver", "description": "supports cells", "rank": 1}]


def _seed(c, email, scan_date=None):
    body = {"email": email, "name": "X",
            "content": {"greeting": "Aloha", "layers": [_LAYER], "findings": []}}
    if scan_date:
        body["scan_date"] = scan_date
    r = c.post("/api/console/biofield-portal?key=test-secret", json=body)
    assert r.status_code == 200
    return r.get_json()["token"]


def test_backfill_requires_console_key(client):
    c, _ = client
    r = c.post("/api/console/portal/backfill-findings", json={"email": "a@b.com", "findings": []})
    assert r.status_code == 401


def test_backfill_unknown_email_404_and_no_create(client):
    c, appmod = client
    import sqlite3
    r = c.post("/api/console/portal/backfill-findings?key=test-secret",
               json={"email": "ghost@none.com", "findings": _F})
    assert r.status_code == 404
    assert r.get_json()["found"] is False
    with sqlite3.connect(appmod.LOG_DB) as cx:
        n = cx.execute("SELECT COUNT(*) FROM client_portals WHERE email=?",
                       ("ghost@none.com",)).fetchone()[0]
    assert n == 0  # never created


def test_backfill_patches_portal_record_findings_only(client):
    c, _ = client
    tok = _seed(c, "rec@b.com")  # no scan_date -> no report row -> portal-record path
    r = c.post("/api/console/portal/backfill-findings?key=test-secret",
               json={"email": "rec@b.com", "findings": _F})
    assert r.status_code == 200
    j = r.get_json()
    assert j["patched_portal"] is True and j["patched_reports"] == 0
    d = c.get(f"/api/portal/{tok}").get_json()
    assert d["findings"] == _F
    assert d["layers"][0]["title"] == "Calm"  # every other field intact


def test_backfill_patches_report_by_scan_date(client):
    c, _ = client
    tok = _seed(c, "rep@b.com", scan_date="2026-06-25")
    f2 = [{"code": "ET1", "name": "Heart Driver", "description": "h", "rank": 1}]
    r = c.post("/api/console/portal/backfill-findings?key=test-secret",
               json={"email": "rep@b.com", "scan_date": "2026-06-25", "findings": f2})
    assert r.status_code == 200
    assert r.get_json()["patched_reports"] == 1
    d = c.get(f"/api/portal/{tok}?scan_date=2026-06-25").get_json()
    assert d["findings"] == f2


def test_backfill_idempotent(client):
    c, _ = client
    tok = _seed(c, "idem@b.com")
    for _ in range(2):
        r = c.post("/api/console/portal/backfill-findings?key=test-secret",
                   json={"email": "idem@b.com", "findings": _F})
        assert r.status_code == 200
    assert c.get(f"/api/portal/{tok}").get_json()["findings"] == _F
