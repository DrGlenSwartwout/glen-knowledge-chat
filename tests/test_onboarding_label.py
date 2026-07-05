import sqlite3
from dashboard import evox as _ev


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _ev.init_evox_tables(cx)
    _ev._ensure_calendar_events(cx) if hasattr(_ev, "_ensure_calendar_events") else None
    cx.execute("""CREATE TABLE IF NOT EXISTS calendar_events (
        id INTEGER PRIMARY KEY, pushed_at TEXT, google_cal_id TEXT, google_event_id TEXT,
        calendar_name TEXT, summary TEXT, start TEXT, end TEXT, location TEXT,
        owner TEXT, status TEXT, cal_alert INTEGER)""")
    return cx


def _label_for(cx, session_type, medium="phone", practitioner="rae"):
    _ev.create_booking(cx, "a@b.com", "2026-07-10T09:00:00", duration_min=15,
                       practitioner=practitioner, session_type=session_type, medium=medium)
    row = cx.execute("SELECT calendar_name, summary FROM calendar_events "
                     "ORDER BY id DESC LIMIT 1").fetchone()
    return row["calendar_name"], row["summary"]


def test_onboarding_label():
    name, summary = _label_for(_cx(), "onboarding")
    assert name == "Welcome Call booking"
    assert summary.startswith("Welcome Call — ")


def test_triage_label():
    name, summary = _label_for(_cx(), "triage")
    assert name == "Discovery Call booking"


def test_consult_label_unchanged():
    name, _ = _label_for(_cx(), "biofield-consult", medium="video", practitioner="glen")
    assert name == "Biofield Consult booking"


def test_evox_label_unchanged():
    name, _ = _label_for(_cx(), "evox")
    assert name == "EVOX booking"
