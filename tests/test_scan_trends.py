import sqlite3
import pytest
from dashboard import scan_trends as st


@pytest.fixture
def cx():
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    c.executescript(
        "CREATE TABLE e4l_scans (scan_id INTEGER PRIMARY KEY, client_id INTEGER, scan_date TEXT);"
        "CREATE TABLE e4l_scan_results (scan_id INTEGER, item_code TEXT, priority_rank INTEGER);"
        "CREATE TABLE e4l_items (code TEXT PRIMARY KEY, name TEXT, full_name TEXT, category TEXT);"
    )
    # 5 scans, oldest(1) -> newest(5). Newer half = scans 4,5; older half = 1,2 (3 is the midpoint).
    c.executemany("INSERT INTO e4l_scans VALUES (?,?,?)", [
        (1, 7, "2026-01-01"), (2, 7, "2026-02-01"), (3, 7, "2026-03-01"),
        (4, 7, "2026-04-01"), (5, 7, "2026-05-01")])
    c.executemany("INSERT INTO e4l_items VALUES (?,?,?,?)", [
        ("ED1", "Source", "Source Driver", "ED"),
        ("ER5", "Old", "Old", "ER"),
        ("EI2", "New", "New", "EI"),
        ("ES1", "Flick", "Flicker", "ES")])
    c.executemany("INSERT INTO e4l_scan_results VALUES (?,?,?)", [
        # ED1 in 3 of 5 scans -> persistent (>=50%)
        (1, "ED1", 2), (3, "ED1", 1), (5, "ED1", 1),
        # ER5 only in the two oldest -> resolving
        (1, "ER5", 5), (2, "ER5", 4),
        # EI2 only in the two newest -> emerging
        (4, "EI2", 3), (5, "EI2", 2),
        # ES1 once in old (2) and once in new (4), 40% -> intermittent
        (2, "ES1", 6), (4, "ES1", 7)])
    c.commit()
    return c


def test_no_scans_is_empty(cx):
    assert st.client_trends(cx, 999)["items"] == []
    assert st.client_trends(cx, 999)["n_scans"] == 0


def test_trend_classification(cx):
    out = st.client_trends(cx, 7)
    assert out["n_scans"] == 5 and out["first_date"] == "2026-01-01" and out["last_date"] == "2026-05-01"
    by = {it["code"]: it for it in out["items"]}
    assert by["ED1"]["trend"] == "persistent" and by["ED1"]["frequency_pct"] == 60
    assert by["ED1"]["best_rank"] == 1 and by["ED1"]["latest_rank"] == 1
    assert by["ER5"]["trend"] == "resolving"
    assert by["EI2"]["trend"] == "emerging"
    assert by["ES1"]["trend"] == "intermittent"
    # persistent sorts first
    assert out["items"][0]["code"] == "ED1"


def test_last_n_limits_window(cx):
    # last 2 scans (4,5): ED1, EI2, ES1 present; ER5 (only in oldest) absent
    out = st.client_trends(cx, 7, last_n=2)
    assert out["n_scans"] == 2
    codes = {it["code"] for it in out["items"]}
    assert "ER5" not in codes and "ED1" in codes


# --- severity (rank normalized within each scan; 1.0 = rank 1 = "purple"/top) ---

def test_severity_normalization():
    assert st._severity(None, 5) is None
    assert st._severity(1, 5) == 1.0        # top of the list = most severe
    assert st._severity(5, 5) == 0.0        # bottom of the list = least severe
    assert st._severity(3, 5) == 0.5        # middle
    assert st._severity(1, 1) == 1.0        # a lone ranked item is top by definition
    # rank beyond the ranked count clamps to 0, never negative
    assert st._severity(9, 5) == 0.0


def test_severity_bands():
    assert st._severity_band(0.90) == "top"
    assert st._severity_band(0.60) == "high"
    assert st._severity_band(0.30) == "moderate"
    assert st._severity_band(0.10) == "low"
    assert st._severity_band(None) is None


def test_severity_trend_helper():
    assert st._severity_trend(0.80, 0.20) == "worsening"   # climbing toward the top
    assert st._severity_trend(0.20, 0.80) == "easing"      # dropping down the list
    assert st._severity_trend(0.50, 0.50) == "steady"
    assert st._severity_trend(None, 0.50) == "na"          # can't compare one half


@pytest.fixture
def sev_cx():
    """4 scans, oldest(1)->newest(5), each with exactly 5 ranked items (ranks 1-5),
    so severity denominators are clean. Three tracked items:
      WORSEN climbs 5->4->2->1 (severity up), EASE drops 1->2->4->5, STEADY sits at 3."""
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    c.executescript(
        "CREATE TABLE e4l_scans (scan_id INTEGER PRIMARY KEY, client_id INTEGER, scan_date TEXT);"
        "CREATE TABLE e4l_scan_results (scan_id INTEGER, item_code TEXT, priority_rank INTEGER);"
        "CREATE TABLE e4l_items (code TEXT PRIMARY KEY, name TEXT, full_name TEXT, category TEXT);"
    )
    c.executemany("INSERT INTO e4l_scans VALUES (?,?,?)", [
        (1, 7, "2026-01-01"), (2, 7, "2026-02-01"),
        (3, 7, "2026-03-01"), (4, 7, "2026-04-01")])
    c.executemany("INSERT INTO e4l_items VALUES (?,?,?,?)", [
        ("WORSEN", "W", "W", "X"), ("EASE", "E", "E", "X"), ("STEADY", "S", "S", "X"),
        ("FZ1", "F", "F", "X"), ("FZ2", "G", "G", "X")])
    # each scan uses ranks {1,2,3,4,5} exactly once -> n_ranked = 5 per scan
    c.executemany("INSERT INTO e4l_scan_results VALUES (?,?,?)", [
        (1, "WORSEN", 5), (1, "EASE", 1), (1, "STEADY", 3), (1, "FZ1", 2), (1, "FZ2", 4),
        (2, "WORSEN", 4), (2, "EASE", 2), (2, "STEADY", 3), (2, "FZ1", 1), (2, "FZ2", 5),
        (3, "WORSEN", 2), (3, "EASE", 4), (3, "STEADY", 3), (3, "FZ1", 1), (3, "FZ2", 5),
        (4, "WORSEN", 1), (4, "EASE", 5), (4, "STEADY", 3), (4, "FZ1", 2), (4, "FZ2", 4)])
    c.commit()
    return c


def test_client_trends_severity_fields(sev_cx):
    by = {it["code"]: it for it in st.client_trends(sev_cx, 7)["items"]}
    # WORSEN: latest appearance is rank 1 -> severity 1.0, band top, climbing
    assert by["WORSEN"]["severity"] == 1.0
    assert by["WORSEN"]["severity_band"] == "top"
    assert by["WORSEN"]["severity_trend"] == "worsening"
    # EASE: latest appearance rank 5 -> severity 0.0, band low, dropping
    assert by["EASE"]["severity"] == 0.0
    assert by["EASE"]["severity_band"] == "low"
    assert by["EASE"]["severity_trend"] == "easing"
    # STEADY: rank 3 throughout -> severity 0.5, band high, no movement
    assert by["STEADY"]["severity"] == 0.5
    assert by["STEADY"]["severity_trend"] == "steady"
