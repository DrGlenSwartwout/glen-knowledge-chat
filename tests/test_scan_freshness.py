import sqlite3
from dashboard import scan_freshness as sf

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    sf.init_table(cx); return cx

def test_upsert_and_latest():
    cx = _cx()
    sf.upsert(cx, [{"email": "p@x.com", "last_scan_date": "2026-06-10"},
                   {"email": "q@x.com", "last_scan_date": "2026-01-01"}])
    assert sf.latest_scan_date(cx, "P@X.COM") == "2026-06-10"
    assert sf.latest_scan_date(cx, "q@x.com") == "2026-01-01"
    assert sf.latest_scan_date(cx, "none@x.com") is None

def test_upsert_keeps_newest():
    cx = _cx()
    sf.upsert(cx, [{"email": "p@x.com", "last_scan_date": "2026-01-01"}])
    sf.upsert(cx, [{"email": "p@x.com", "last_scan_date": "2026-06-10"}])
    assert sf.latest_scan_date(cx, "p@x.com") == "2026-06-10"
    sf.upsert(cx, [{"email": "p@x.com", "last_scan_date": "2026-03-01"}])  # older, must not regress
    assert sf.latest_scan_date(cx, "p@x.com") == "2026-06-10"

def test_is_fresh():
    cx = _cx()
    sf.upsert(cx, [{"email": "p@x.com", "last_scan_date": "2026-06-12"}])
    assert sf.is_fresh(cx, "p@x.com", today="2026-06-15", window_days=7) is True
    assert sf.is_fresh(cx, "p@x.com", today="2026-06-25", window_days=7) is False
    assert sf.is_fresh(cx, "none@x.com", today="2026-06-15", window_days=7) is False

def test_upsert_skips_blank():
    cx = _cx()
    sf.upsert(cx, [{"email": "", "last_scan_date": "2026-06-10"},
                   {"email": "p@x.com", "last_scan_date": ""}])
    assert sf.latest_scan_date(cx, "p@x.com") is None
