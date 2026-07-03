import sqlite3
import app as appmod
from dashboard import family_access as fa
from dashboard import portal_biofield_reports as pbr


def _seed(tmp_db):
    cx = sqlite3.connect(tmp_db)
    fa.init_tables(cx); pbr.init_table(cx)
    pbr.upsert_report(cx, "m@x.com", "2026-07-02", "s1",
                      {"layers": [{"n": 1, "title": "T", "meaning": "M", "remedy": "R", "dosing": "D"}]},
                      "confirmed")
    cx.commit(); cx.close()


def test_v2_locked_scan_hides_remedy(tmp_db, monkeypatch):
    monkeypatch.setenv("PORTAL_ACCESS_V2", "1")
    monkeypatch.setattr(appmod, "LOG_DB", tmp_db)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: False)
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "m@x.com", "name": "M"})
    _seed(tmp_db)
    j = appmod.app.test_client().get("/api/portal/TOK").get_json()
    assert j["blurred"] is True
    assert j["layers"][0].get("remedy", "") == ""   # locked -> no remedy


def test_v2_paid_shows_remedy(tmp_db, monkeypatch):
    monkeypatch.setenv("PORTAL_ACCESS_V2", "1")
    monkeypatch.setattr(appmod, "LOG_DB", tmp_db)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: True)
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "m@x.com", "name": "M"})
    _seed(tmp_db)
    j = appmod.app.test_client().get("/api/portal/TOK").get_json()
    assert j["blurred"] is False
    assert j["layers"][0]["remedy"] == "R"


def test_access_v2_flag_reflected_in_response(tmp_db, monkeypatch):
    monkeypatch.setattr(appmod, "LOG_DB", tmp_db)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: False)
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "m@x.com", "name": "M"})
    _seed(tmp_db)

    monkeypatch.setenv("PORTAL_ACCESS_V2", "1")
    j = appmod.app.test_client().get("/api/portal/TOK").get_json()
    assert j["access_v2"] is True

    monkeypatch.delenv("PORTAL_ACCESS_V2", raising=False)
    j = appmod.app.test_client().get("/api/portal/TOK").get_json()
    assert j["access_v2"] is False
