import sqlite3
from dashboard import client_scans as cs


def _cx():
    cx = sqlite3.connect(":memory:"); cs.init_client_scans_table(cx); return cx


def test_upsert_and_list():
    cx = _cx()
    n = cs.upsert_scans(cx, "Karin@X.com", [{"scan_date": "2026-06-28", "scan_id": 1037676},
                                            {"scan_date": "2026-06-25", "scan_id": 1037001}])
    assert n == 2
    got = cs.scans_for(cx, "karin@x.com")
    assert [g["scan_date"] for g in got] == ["2026-06-28", "2026-06-25"]   # most-recent first
    assert got[0]["scan_id"] == "1037676"                                  # stored as str


def test_upsert_idempotent():
    cx = _cx()
    cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-06-28", "scan_id": 1}])
    cs.upsert_scans(cx, "k@x.com", [{"scan_date": "2026-06-28", "scan_id": 2}])   # same date, new id
    got = cs.scans_for(cx, "k@x.com")
    assert len(got) == 1 and got[0]["scan_id"] == "2"                     # no dup; scan_id updated


def test_blank_email_and_date_skipped():
    cx = _cx()
    assert cs.upsert_scans(cx, "", [{"scan_date": "2026-06-28"}]) == 0
    assert cs.upsert_scans(cx, "k@x.com", [{"scan_date": ""}]) == 0
    assert cs.scans_for(cx, "k@x.com") == []
