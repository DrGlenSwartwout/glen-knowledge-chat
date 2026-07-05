"""MasterClass events + registrations. Stdlib-only; import without importing app."""
import sqlite3
from datetime import datetime, timezone

def _now():
    return datetime.now(timezone.utc).isoformat()

def init_masterclass_tables(cx) -> None:
    cx.execute("""CREATE TABLE IF NOT EXISTS masterclass_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, description TEXT,
        start_ts TEXT, duration_min INTEGER DEFAULT 60,
        price_cents INTEGER DEFAULT 0, member_price_cents INTEGER DEFAULT 0,
        zoom_join_url TEXT, zoom_meeting_id TEXT, created_at TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS masterclass_registrations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER, email TEXT, name TEXT,
        is_member INTEGER, amount_cents INTEGER, paid INTEGER DEFAULT 0, created_at TEXT,
        UNIQUE(event_id, email))""")
    cx.commit()

def create_event(cx, *, topic, description, start_ts, duration_min,
                 price_cents, member_price_cents) -> int:
    cur = cx.execute(
        "INSERT INTO masterclass_events (topic, description, start_ts, duration_min, "
        "price_cents, member_price_cents, created_at) VALUES (?,?,?,?,?,?,?)",
        ((topic or "").strip(), (description or "").strip(), start_ts, int(duration_min),
         int(price_cents), int(member_price_cents), _now()))
    cx.commit()
    return cur.lastrowid

def get_event(cx, event_id):
    cur = cx.execute("SELECT * FROM masterclass_events WHERE id=?", (event_id,))
    cols = [c[0] for c in cur.description]; r = cur.fetchone()
    return dict(zip(cols, r)) if r is not None else None

def set_zoom(cx, event_id, join_url, meeting_id) -> None:
    cx.execute("UPDATE masterclass_events SET zoom_join_url=?, zoom_meeting_id=? WHERE id=?",
               (join_url, meeting_id, event_id))
    cx.commit()

def price_for(event, is_member) -> int:
    return int(event["member_price_cents"] if is_member else event["price_cents"])

def register(cx, event_id, email, name, is_member, amount_cents, *, paid) -> None:
    email = (email or "").strip().lower()
    cx.execute("INSERT INTO masterclass_registrations (event_id, email, name, is_member, "
               "amount_cents, paid, created_at) VALUES (?,?,?,?,?,?,?) "
               "ON CONFLICT(event_id, email) DO UPDATE SET name=excluded.name, "
               "is_member=excluded.is_member, amount_cents=excluded.amount_cents, "
               "paid=excluded.paid",
               (event_id, email, (name or "").strip(), 1 if is_member else 0,
                int(amount_cents), 1 if paid else 0, _now()))
    cx.commit()

def mark_paid(cx, event_id, email) -> None:
    cx.execute("UPDATE masterclass_registrations SET paid=1 WHERE event_id=? AND lower(email)=?",
               (event_id, (email or "").strip().lower()))
    cx.commit()

def is_registered(cx, event_id, email) -> bool:
    r = cx.execute("SELECT 1 FROM masterclass_registrations WHERE event_id=? AND lower(email)=? "
                   "AND paid=1", (event_id, (email or "").strip().lower())).fetchone()
    return r is not None
