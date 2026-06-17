import sqlite3, datetime
from dashboard import portal_biofield_reports as R


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db")); R.init_table(cx); return cx


def test_upsert_get_and_overwrite(tmp_path):
    cx = _cx(tmp_path)
    R.upsert_report(cx, "a@x.com", "2026-06-05", "s1", {"layers": [{"n": 1, "title": "t"}]}, "ai_draft")
    rep = R.get_report(cx, "a@x.com", "2026-06-05")
    assert rep["status"] == "ai_draft" and rep["scan_id"] == "s1"
    assert rep["content"]["layers"][0]["title"] == "t"
    R.upsert_report(cx, "a@x.com", "2026-06-05", "s1", {"layers": []}, "confirmed")
    assert R.get_report(cx, "a@x.com", "2026-06-05")["status"] == "confirmed"
    assert R.list_report_dates(cx, "a@x.com") == ["2026-06-05"]


def test_list_dates_newest_first_and_latest(tmp_path):
    cx = _cx(tmp_path)
    for d in ["2026-04-01", "2026-06-05", "2026-05-02"]:
        R.upsert_report(cx, "a@x.com", d, "s", {"layers": []}, "ai_draft")
    assert R.list_report_dates(cx, "a@x.com") == ["2026-06-05", "2026-05-02", "2026-04-01"]
    assert R.latest_report(cx, "a@x.com")["scan_date"] == "2026-06-05"
    assert R.get_report(cx, "a@x.com", "nope") is None
    assert R.latest_report(cx, "nobody@x.com") is None


def test_set_status(tmp_path):
    cx = _cx(tmp_path)
    R.upsert_report(cx, "a@x.com", "2026-06-05", "s", {"layers": []}, "ai_draft")
    assert R.set_report_status(cx, "a@x.com", "2026-06-05", "requested") is True
    assert R.get_report(cx, "a@x.com", "2026-06-05")["status"] == "requested"
    assert R.set_report_status(cx, "a@x.com", "missing", "requested") is False


def test_is_actionable_window():
    today = "2026-06-17"
    assert R.is_actionable("2026-06-05", today) is True     # within 30 days
    assert R.is_actionable("2026-06-17", today) is True     # today
    assert R.is_actionable("2026-05-18", today) is True     # exactly 30 days -> inclusive
    assert R.is_actionable("2026-05-17", today) is False    # 31 days -> out
    assert R.is_actionable("2026-04-01", today) is False    # > 30 days
    assert R.is_actionable("", today) is False              # no/garbage date
