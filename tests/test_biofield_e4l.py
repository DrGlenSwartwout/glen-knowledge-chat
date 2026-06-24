"""E4L scan pull for the local Biofield Analysis tool: scan_context() reads the
separate ~/AI-Training/e4l.db (read-only), reports freshness + days-ago, and returns
the most recent scan's ranked findings. Never raises; missing DB/client -> status none."""
import sqlite3
import pytest

from dashboard.biofield_e4l import scan_context


def _seed(path):
    """Minimal e4l.db: clients, scans, results, items (+ identity_merges)."""
    cx = sqlite3.connect(str(path))
    cx.executescript("""
        CREATE TABLE e4l_clients(client_id INTEGER PRIMARY KEY, name TEXT, email TEXT);
        CREATE TABLE e4l_scans(scan_id INTEGER PRIMARY KEY, client_id INTEGER,
                               scan_date TEXT NOT NULL);
        CREATE TABLE e4l_scan_results(id INTEGER PRIMARY KEY AUTOINCREMENT, scan_id INTEGER,
                               item_code TEXT, priority_rank INTEGER);
        CREATE TABLE e4l_items(code TEXT PRIMARY KEY, name TEXT, full_name TEXT,
                               category TEXT, e4l_description TEXT, clinical_notes TEXT);
    """)
    cx.execute("INSERT INTO e4l_clients VALUES(100,'Jane Doe','jane@x.com')")
    cx.executemany("INSERT INTO e4l_items(code,name,full_name,e4l_description) VALUES(?,?,?,?)",
                   [("LV3", "Liver", "Liver meridian", "detox and anger"),
                    ("KI1", "Kidney", "Kidney meridian", "fear and adrenal")])
    return cx


def test_fresh_scan_within_window(tmp_path):
    db = tmp_path / "e4l.db"
    cx = _seed(db)
    cx.execute("INSERT INTO e4l_scans VALUES(900,100,'2026-06-20')")
    cx.execute("INSERT INTO e4l_scan_results(scan_id,item_code,priority_rank) VALUES(900,'LV3',1)")
    cx.execute("INSERT INTO e4l_scan_results(scan_id,item_code,priority_rank) VALUES(900,'KI1',2)")
    cx.commit(); cx.close()
    ctx = scan_context("jane@x.com", "2026-06-24", db_path=str(db))
    assert ctx["status"] == "fresh"
    assert ctx["found"] is True and ctx["fresh"] is True
    assert ctx["scan_id"] == 900 and ctx["scan_date"] == "2026-06-20"
    assert ctx["days_ago"] == 4
    # findings ranked, with readable name + description
    assert [f["code"] for f in ctx["findings"]] == ["LV3", "KI1"]
    assert ctx["findings"][0]["name"] == "Liver meridian"
    assert ctx["findings"][0]["rank"] == 1
    assert "detox" in ctx["findings"][0]["description"]


def test_stale_scan_outside_window_still_returns_findings(tmp_path):
    db = tmp_path / "e4l.db"
    cx = _seed(db)
    cx.execute("INSERT INTO e4l_scans VALUES(800,100,'2026-05-17')")
    cx.execute("INSERT INTO e4l_scan_results(scan_id,item_code,priority_rank) VALUES(800,'LV3',1)")
    cx.commit(); cx.close()
    ctx = scan_context("jane@x.com", "2026-06-24", db_path=str(db))
    assert ctx["status"] == "stale"
    assert ctx["found"] is True and ctx["fresh"] is False
    assert ctx["days_ago"] == 38
    assert [f["code"] for f in ctx["findings"]] == ["LV3"]  # still shown


def test_picks_most_recent_scan(tmp_path):
    db = tmp_path / "e4l.db"
    cx = _seed(db)
    cx.execute("INSERT INTO e4l_scans VALUES(700,100,'2026-01-01')")
    cx.execute("INSERT INTO e4l_scans VALUES(701,100,'2026-06-22')")
    cx.commit(); cx.close()
    ctx = scan_context("jane@x.com", "2026-06-24", db_path=str(db))
    assert ctx["scan_id"] == 701 and ctx["days_ago"] == 2


def test_window_boundary_is_inclusive(tmp_path):
    db = tmp_path / "e4l.db"
    cx = _seed(db)
    cx.execute("INSERT INTO e4l_scans VALUES(600,100,'2026-06-10')")  # exactly 14 days
    cx.commit(); cx.close()
    ctx = scan_context("jane@x.com", "2026-06-24", db_path=str(db))
    assert ctx["days_ago"] == 14 and ctx["fresh"] is True and ctx["status"] == "fresh"


def test_future_scan_date_clamps_days_ago_to_zero(tmp_path):
    db = tmp_path / "e4l.db"
    cx = _seed(db)
    cx.execute("INSERT INTO e4l_scans VALUES(500,100,'2026-06-30')")  # data glitch: future
    cx.commit(); cx.close()
    ctx = scan_context("jane@x.com", "2026-06-24", db_path=str(db))
    assert ctx["days_ago"] == 0 and ctx["fresh"] is True


def test_no_scan_for_known_client(tmp_path):
    db = tmp_path / "e4l.db"
    cx = _seed(db); cx.commit(); cx.close()  # client exists, no scans
    ctx = scan_context("jane@x.com", "2026-06-24", db_path=str(db))
    assert ctx["status"] == "none" and ctx["found"] is False
    assert ctx["findings"] == [] and ctx["days_ago"] is None


def test_unknown_email(tmp_path):
    db = tmp_path / "e4l.db"
    cx = _seed(db); cx.commit(); cx.close()
    ctx = scan_context("nobody@x.com", "2026-06-24", db_path=str(db))
    assert ctx["status"] == "none" and ctx["found"] is False


def test_blank_email(tmp_path):
    db = tmp_path / "e4l.db"
    cx = _seed(db); cx.commit(); cx.close()
    assert scan_context("", "2026-06-24", db_path=str(db))["status"] == "none"
    assert scan_context(None, "2026-06-24", db_path=str(db))["status"] == "none"


def test_missing_db_returns_none_not_raises(tmp_path):
    ctx = scan_context("jane@x.com", "2026-06-24", db_path=str(tmp_path / "nope.db"))
    assert ctx["status"] == "none" and ctx["found"] is False


def test_email_match_is_case_insensitive(tmp_path):
    db = tmp_path / "e4l.db"
    cx = _seed(db)
    cx.execute("INSERT INTO e4l_scans VALUES(900,100,'2026-06-22')")
    cx.commit(); cx.close()
    ctx = scan_context("JANE@X.com", "2026-06-24", db_path=str(db))
    assert ctx["found"] is True and ctx["scan_id"] == 900


def test_identity_merge_reads_split_history(tmp_path):
    """A duplicate account's scan is found under the canonical email."""
    db = tmp_path / "e4l.db"
    cx = _seed(db)
    cx.execute("CREATE TABLE e4l_identity_merges(dup_client_id INTEGER PRIMARY KEY,"
               "canonical_client_id INTEGER NOT NULL, note TEXT, confirmed_at TEXT)")
    # second account (dup) for the same person, with the more recent scan
    cx.execute("INSERT INTO e4l_clients VALUES(101,'Jane D','jane.alt@x.com')")
    cx.execute("INSERT INTO e4l_identity_merges VALUES(101,100,'same person','2026-06-01')")
    cx.execute("INSERT INTO e4l_scans VALUES(900,100,'2026-06-01')")  # canonical, older
    cx.execute("INSERT INTO e4l_scans VALUES(950,101,'2026-06-23')")  # dup, newer
    cx.commit(); cx.close()
    ctx = scan_context("jane@x.com", "2026-06-24", db_path=str(db))
    assert ctx["scan_id"] == 950 and ctx["days_ago"] == 1


def test_limit_caps_findings(tmp_path):
    db = tmp_path / "e4l.db"
    cx = _seed(db)
    cx.execute("INSERT INTO e4l_scans VALUES(900,100,'2026-06-22')")
    for i in range(20):
        cx.execute("INSERT INTO e4l_items(code,name) VALUES(?,?)", (f"X{i}", f"item{i}"))
        cx.execute("INSERT INTO e4l_scan_results(scan_id,item_code,priority_rank) "
                   "VALUES(900,?,?)", (f"X{i}", i + 1))
    cx.commit(); cx.close()
    ctx = scan_context("jane@x.com", "2026-06-24", db_path=str(db), limit=5)
    assert len(ctx["findings"]) == 5
