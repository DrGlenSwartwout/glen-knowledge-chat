# tests/test_family_unlock_endpoint.py
import json, sqlite3
import app as appmod


def _client(tmp_db, monkeypatch):
    monkeypatch.setattr(appmod, "LOG_DB", tmp_db)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: False)
    # a portal token that resolves to member m@x.com
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "m@x.com"} if tok == "TOK" else None)
    return appmod.app.test_client()


def test_unlock_scan_first_succeeds_then_capped(tmp_db, monkeypatch):
    c = _client(tmp_db, monkeypatch)
    r1 = c.post("/api/portal/TOK/unlock-scan", json={"scan_id": "s1"})
    assert r1.status_code == 200 and r1.get_json()["ok"] is True
    r2 = c.post("/api/portal/TOK/unlock-scan", json={"scan_id": "s2"})
    assert r2.get_json() == {"ok": False, "reason": "cap"}


def test_unlock_scan_unknown_token_404(tmp_db, monkeypatch):
    c = _client(tmp_db, monkeypatch)
    r = c.post("/api/portal/NOPE/unlock-scan", json={"scan_id": "s1"})
    assert r.status_code == 404
