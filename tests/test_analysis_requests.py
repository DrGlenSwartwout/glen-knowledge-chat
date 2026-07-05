import sqlite3
from dashboard import analysis_requests as ar


def _cx():
    cx = sqlite3.connect(":memory:"); ar.init_analysis_requests_table(cx); return cx


def test_create_idempotent():
    cx = _cx()
    assert ar.create_request(cx, "K@x.com", 7, "2026-06-28") == {"created": True, "status": "pending"}
    assert ar.create_request(cx, "k@x.com", 7, "2026-06-28") == {"created": False, "status": "pending"}
    assert ar.has_pending(cx, "k@x.com", "2026-06-28") is True


def test_pending_and_mark_and_statuses():
    cx = _cx()
    ar.create_request(cx, "a@x.com", 1, "2026-06-01")
    ar.create_request(cx, "b@x.com", 2, "2026-06-02")
    p = ar.pending(cx)
    assert {r["email"] for r in p} == {"a@x.com", "b@x.com"}
    ar.mark(cx, p[0]["id"], "done")
    assert ar.has_pending(cx, p[0]["email"], p[0]["scan_date"]) is False
    assert ar.statuses_for(cx, p[0]["email"])[p[0]["scan_date"]] == "done"


def test_blank_skipped():
    cx = _cx()
    assert ar.create_request(cx, "", 1, "2026-06-01")["created"] is False
    assert ar.create_request(cx, "a@x.com", 1, "")["created"] is False
