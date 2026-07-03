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


def test_v2_member_report_rendered(tmp_db, monkeypatch):
    # primary p@x.com has a report; member m@x.com (in p's family) has a
    # DIFFERENT report. The token belongs to p@x.com; requesting
    # ?member=m@x.com must render M's report content, not P's.
    monkeypatch.setenv("PORTAL_ACCESS_V2", "1")
    monkeypatch.setattr(appmod, "LOG_DB", tmp_db)
    monkeypatch.setattr(appmod, "_is_paid_member", lambda e: True)
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "p@x.com", "name": "P"})
    cx = sqlite3.connect(tmp_db)
    fa.init_tables(cx); pbr.init_table(cx)
    fa.upsert_member(cx, "p@x.com", "p@x.com", "P", "human", 0)
    fa.upsert_member(cx, "p@x.com", "m@x.com", "M", "human", 1)
    pbr.upsert_report(cx, "p@x.com", "2026-07-01", "sP",
                      {"layers": [{"n": 1, "title": "PrimaryReport", "meaning": "M", "remedy": "RP", "dosing": "D"}]},
                      "confirmed")
    pbr.upsert_report(cx, "m@x.com", "2026-07-02", "sM",
                      {"layers": [{"n": 1, "title": "MemberReport", "meaning": "M", "remedy": "RM", "dosing": "D"}]},
                      "confirmed")
    cx.commit(); cx.close()

    j = appmod.app.test_client().get("/api/portal/TOK?member=m@x.com").get_json()
    assert j["scan_date"] == "2026-07-02"
    assert j["layers"][0]["title"] == "MemberReport"
    assert j["layers"][0]["remedy"] == "RM"
    assert [m["member_email"] for m in j["members"]] == ["p@x.com", "m@x.com"]


def test_v2_unauthorized_member_falls_back(tmp_db, monkeypatch):
    # token owner p@x.com is not a family primary of the "stranger" — the
    # request must silently fall back to p's own report, never leaking the
    # stranger's report or letting the stranger's paid status un-blur it.
    monkeypatch.setenv("PORTAL_ACCESS_V2", "1")
    monkeypatch.setattr(appmod, "LOG_DB", tmp_db)

    def _paid(email):
        return email == "stranger@x.com"  # only the stranger is "paid"
    monkeypatch.setattr(appmod, "_is_paid_member", _paid)
    monkeypatch.setattr(appmod, "_portal_record_for", lambda cx, tok: {"email": "p@x.com", "name": "P"})
    cx = sqlite3.connect(tmp_db)
    fa.init_tables(cx); pbr.init_table(cx)
    pbr.upsert_report(cx, "p@x.com", "2026-07-01", "sP",
                      {"layers": [{"n": 1, "title": "PrimaryReport", "meaning": "M", "remedy": "RP", "dosing": "D"}]},
                      "confirmed")
    pbr.upsert_report(cx, "stranger@x.com", "2026-07-05", "sS",
                      {"layers": [{"n": 1, "title": "StrangerReport", "meaning": "M", "remedy": "RS", "dosing": "D"}]},
                      "confirmed")
    cx.commit(); cx.close()

    j = appmod.app.test_client().get("/api/portal/TOK?member=stranger@x.com").get_json()
    assert j["scan_date"] == "2026-07-01"
    assert j["layers"][0]["title"] == "PrimaryReport"
    # p@x.com is not paid, so the report stays blurred even though the
    # stranger (whose report we did NOT render) is "paid"
    assert j["blurred"] is True
    assert j["layers"][0].get("remedy", "") == ""
