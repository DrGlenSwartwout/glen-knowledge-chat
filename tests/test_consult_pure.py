import sqlite3
import json
from dashboard import evox, consult

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    evox.init_evox_tables(cx)
    cx.execute("""CREATE TABLE calendar_events (id INTEGER PRIMARY KEY AUTOINCREMENT,
        pushed_at TEXT, google_cal_id TEXT, google_event_id TEXT, calendar_name TEXT,
        summary TEXT, start TEXT, end TEXT, location TEXT, owner TEXT, status TEXT,
        cal_alert INTEGER, UNIQUE(google_cal_id, google_event_id))""")
    cx.commit(); return cx

def test_consult_booking_sets_type_medium_and_video_calendar_row():
    cx = _cx()
    b = evox.create_booking(cx, "c@x.com", "2026-07-06T13:00:00",
                            duration_min=30, practitioner="glen",
                            session_type="biofield-consult", medium="video")
    assert b["end_ts"] == "2026-07-06T13:30:00"
    assert b["session_type"] == "biofield-consult" and b["medium"] == "video"
    row = cx.execute("SELECT owner, location, summary, session_type, medium "
                     "FROM evox_bookings JOIN calendar_events "
                     "ON calendar_events.google_event_id='biofield-consult-'||evox_bookings.id").fetchone()
    assert row["owner"] == "glen" and row["location"] == "Video"
    assert "Biofield Consult" in row["summary"]
    assert row["session_type"] == "biofield-consult" and row["medium"] == "video"

def test_evox_booking_defaults_unchanged():
    cx = _cx()
    b = evox.create_booking(cx, "e@x.com", "2026-07-06T11:00:00")
    assert b["session_type"] == "evox" and b["medium"] == "phone"
    row = cx.execute("SELECT location, summary FROM calendar_events").fetchone()
    assert row["location"] == "Phone" and row["summary"] == "EVOX — e@x.com"

def _ccx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    consult.init_consult_tables(cx); return cx

def test_consult_ready_roundtrip():
    cx = _ccx()
    assert consult.consult_is_ready(cx, "A@x.com") is False
    assert consult.set_consult_ready(cx, "a@x.com", True) is True
    assert consult.consult_is_ready(cx, "A@x.com") is True      # lowercased
    assert consult.set_consult_ready(cx, "a@x.com", False) is False
    assert consult.consult_is_ready(cx, "a@x.com") is False

def test_has_paid_purchase():
    cx = _ccx()
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, email TEXT, items_json TEXT, "
               "pay_status TEXT, paid_cents INTEGER)")
    cx.execute("INSERT INTO orders (email, items_json, pay_status, paid_cents) VALUES (?,?,?,?)",
               ("buyer@x.com", json.dumps([{"slug": "biofield-analysis"}]), "paid", 30000))
    cx.execute("INSERT INTO orders (email, items_json, pay_status, paid_cents) VALUES (?,?,?,?)",
               ("unpaid@x.com", json.dumps([{"slug": "biofield-analysis"}]), "unpaid", 0))
    cx.commit()
    assert consult.has_paid_purchase(cx, "BUYER@x.com", "biofield-analysis") is True
    assert consult.has_paid_purchase(cx, "unpaid@x.com", "biofield-analysis") is False
    assert consult.has_paid_purchase(cx, "nobody@x.com", "biofield-analysis") is False
