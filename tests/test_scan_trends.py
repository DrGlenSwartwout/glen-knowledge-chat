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
