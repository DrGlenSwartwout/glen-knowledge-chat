import sqlite3
from dashboard import evox as _ev
from dashboard import portal_view as _pv


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _ev.init_evox_tables(cx)
    # init_evox_tables doesn't create calendar_events (that's owned by the
    # calendar module); create_booking pushes a row there too, so tests that
    # book need it. Matches the fixture in test_onboarding_pure.py.
    cx.execute("""CREATE TABLE calendar_events (id INTEGER PRIMARY KEY AUTOINCREMENT,
        pushed_at TEXT, google_cal_id TEXT, google_event_id TEXT, calendar_name TEXT,
        summary TEXT, start TEXT, end TEXT, location TEXT, owner TEXT, status TEXT,
        cal_alert INTEGER, UNIQUE(google_cal_id, google_event_id))""")
    cx.commit()
    return cx


def test_block_no_booking():
    cx = _cx()
    b = _pv._onboarding_block(cx, "a@b.com")
    assert b == {"eligible": False, "booked_start": None}


def test_block_with_booking():
    cx = _cx()
    _ev.create_booking(cx, "a@b.com", "2026-07-10T09:00:00", duration_min=15,
                       practitioner="rae", session_type="onboarding", medium="phone")
    b = _pv._onboarding_block(cx, "a@b.com")
    assert b["booked_start"] == "2026-07-10T09:00:00"
    assert b["eligible"] is True


def test_block_never_raises():
    # a broken connection must fall back safely, not raise
    b = _pv._onboarding_block(None, "a@b.com")
    assert b == {"eligible": False, "booked_start": None}
