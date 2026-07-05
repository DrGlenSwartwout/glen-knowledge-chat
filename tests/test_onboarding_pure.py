import sqlite3
from dashboard import evox as _ev
from dashboard import onboarding as _ob


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    _ev.init_evox_tables(cx)
    cx.execute("""CREATE TABLE calendar_events (id INTEGER PRIMARY KEY AUTOINCREMENT,
        pushed_at TEXT, google_cal_id TEXT, google_event_id TEXT, calendar_name TEXT,
        summary TEXT, start TEXT, end TEXT, location TEXT, owner TEXT, status TEXT,
        cal_alert INTEGER, UNIQUE(google_cal_id, google_event_id))""")
    cx.commit()
    return cx


def test_config_values():
    assert _ob.ONBOARDING["session_type"] == "onboarding"
    assert _ob.ONBOARDING["practitioner"] == "rae"
    assert _ob.ONBOARDING["medium"] == "phone"
    assert _ob.ONBOARDING["duration_min"] == 15


def test_existing_none_when_no_booking():
    cx = _cx()
    assert _ob.existing_onboarding(cx, "a@b.com") is None


def test_existing_returns_booked_row():
    cx = _cx()
    _ev.create_booking(cx, "A@B.com", "2026-07-10T09:00:00", duration_min=15,
                       practitioner="rae", session_type="onboarding", medium="phone")
    row = _ob.existing_onboarding(cx, "a@b.com")
    assert row is not None
    assert row["start_ts"] == "2026-07-10T09:00:00"


def test_existing_ignores_other_session_types():
    cx = _cx()
    _ev.create_booking(cx, "a@b.com", "2026-07-10T09:00:00", session_type="evox")
    assert _ob.existing_onboarding(cx, "a@b.com") is None


def test_existing_ignores_cancelled():
    cx = _cx()
    _ev.create_booking(cx, "a@b.com", "2026-07-10T09:00:00", duration_min=15,
                       practitioner="rae", session_type="onboarding", medium="phone")
    cx.execute("UPDATE evox_bookings SET status='cancelled' WHERE lower(email)='a@b.com'")
    cx.commit()
    assert _ob.existing_onboarding(cx, "a@b.com") is None
