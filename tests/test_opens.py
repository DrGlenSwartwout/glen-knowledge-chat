import sqlite3
from dashboard import opens as o


def _cx():
    cx = sqlite3.connect(":memory:"); o.init_opens_table(cx); return cx


def test_keys():
    assert o.report_key(" Karin@X.com ", "2026-06-25") == "karin@x.com|2026-06-25"
    assert o.invoice_key(" tok123 ") == "tok123"


def test_first_open_sets_all():
    cx = _cx()
    r = o.record_open(cx, "report", "k", now="2026-07-04 10:00:00")
    assert r == {"first_opened": "2026-07-04 10:00:00", "last_opened": "2026-07-04 10:00:00", "open_count": 1}


def test_reopen_after_window_bumps_count():
    cx = _cx()
    o.record_open(cx, "report", "k", now="2026-07-04 10:00:00")
    r = o.record_open(cx, "report", "k", now="2026-07-04 10:00:10")   # +10s
    assert r["open_count"] == 2 and r["last_opened"] == "2026-07-04 10:00:10"
    assert r["first_opened"] == "2026-07-04 10:00:00"                 # unchanged


def test_reopen_within_debounce_does_not_bump():
    cx = _cx()
    o.record_open(cx, "report", "k", now="2026-07-04 10:00:00")
    r = o.record_open(cx, "report", "k", now="2026-07-04 10:00:03")   # +3s < 5s
    assert r["open_count"] == 1 and r["last_opened"] == "2026-07-04 10:00:03"


def test_get_and_opens_for_and_clear():
    cx = _cx()
    o.record_open(cx, "invoice", "t1", now="2026-07-04 10:00:00")
    assert o.get_open(cx, "invoice", "t1")["open_count"] == 1
    assert o.get_open(cx, "invoice", "missing") is None
    assert set(o.opens_for(cx, "invoice", ["t1", "missing"]).keys()) == {"t1"}
    o.clear_open(cx, "invoice", "t1")
    assert o.get_open(cx, "invoice", "t1") is None
