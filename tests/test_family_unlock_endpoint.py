# tests/test_family_unlock_endpoint.py
import json, sqlite3
import app as appmod
from dashboard import family_access as fa


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


def test_unlock_scan_already_accessible_no_burn(tmp_db, monkeypatch):
    c = _client(tmp_db, monkeypatch)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: True)
    r1 = c.post("/api/portal/TOK/unlock-scan", json={"scan_id": "s1"})
    assert r1.status_code == 200
    assert r1.get_json() == {"ok": True, "reason": "already"}

    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: False)
    r2 = c.post("/api/portal/TOK/unlock-scan", json={"scan_id": "s2"})
    assert r2.status_code == 200
    assert r2.get_json() == {"ok": True, "reason": ""}


def test_unlock_scan_missing_scan_id_400(tmp_db, monkeypatch):
    c = _client(tmp_db, monkeypatch)
    r = c.post("/api/portal/TOK/unlock-scan", json={})
    assert r.status_code == 400


def test_unlock_rejects_non_family_member(tmp_db, monkeypatch):
    # token's email (m@x.com) is NOT a family primary of stranger@x.com ->
    # the request must be rejected 403, and stranger's allowance must not
    # be burned.
    c = _client(tmp_db, monkeypatch)
    r = c.post("/api/portal/TOK/unlock-scan", json={"member": "stranger@x.com", "scan_id": "s1"})
    assert r.status_code == 403
    assert r.get_json()["ok"] is False
    cx = sqlite3.connect(tmp_db)
    fa.init_tables(cx)
    assert fa.has_unlock(cx, "stranger@x.com", "s1") is False
    cx.close()


def test_unlock_allows_family_member(tmp_db, monkeypatch):
    # P is the token's primary email; M is a real member of P's family.
    monkeypatch.setattr(appmod, "LOG_DB", tmp_db)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: False)
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "p@x.com"} if tok == "PTOK" else None)
    cx = sqlite3.connect(tmp_db)
    fa.init_tables(cx)
    fa.upsert_member(cx, "p@x.com", "p@x.com", "P", "human", 0)
    fa.upsert_member(cx, "p@x.com", "m@x.com", "M", "human", 1)
    cx.commit(); cx.close()

    c = appmod.app.test_client()
    r = c.post("/api/portal/PTOK/unlock-scan", json={"member": "m@x.com", "scan_id": "s1"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True

    cx = sqlite3.connect(tmp_db)
    fa.init_tables(cx)
    assert fa.has_unlock(cx, "m@x.com", "s1") is True
    assert fa.has_unlock(cx, "p@x.com", "s1") is False
    cx.close()
