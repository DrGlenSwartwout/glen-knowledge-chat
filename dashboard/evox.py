"""EVOX booking: self-attest readiness, availability, 1:1 phone booking, ICS.
Pure helpers take primitives only (no cx) and must import without importing app."""
import sqlite3
from datetime import datetime, timedelta

READINESS_ITEMS = ("pc_ok", "cradle_ok", "headset_ok", "zyto_ok")


def readiness_complete(state: dict) -> bool:
    return all(bool(state.get(k)) for k in READINESS_ITEMS)


def init_evox_tables(cx) -> None:
    cx.execute("""CREATE TABLE IF NOT EXISTS evox_readiness (
        email TEXT PRIMARY KEY,
        pc_ok INTEGER DEFAULT 0, cradle_ok INTEGER DEFAULT 0,
        headset_ok INTEGER DEFAULT 0, zyto_ok INTEGER DEFAULT 0,
        cradle_source TEXT, completed_at TEXT, updated_at TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS evox_bookings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL, practitioner TEXT NOT NULL DEFAULT 'rae',
        start_ts TEXT NOT NULL, end_ts TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'booked', prepaid INTEGER DEFAULT 0,
        calendar_event_id TEXT, ics_uid TEXT, created_at TEXT)""")
    cx.execute("""CREATE UNIQUE INDEX IF NOT EXISTS ux_evox_active_slot
        ON evox_bookings(practitioner, start_ts) WHERE status='booked'""")
    cx.execute("""CREATE TABLE IF NOT EXISTS evox_session_credits (
        email TEXT PRIMARY KEY, credits INTEGER NOT NULL DEFAULT 0)""")
    cx.commit()


def get_readiness(cx, email: str) -> dict:
    email = (email or "").strip().lower()
    row = cx.execute("SELECT * FROM evox_readiness WHERE email=?", (email,)).fetchone()
    if row is None:
        base = {k: False for k in READINESS_ITEMS}
        base.update({"email": email, "cradle_source": None, "complete": False})
        return base
    d = dict(row)
    out = {k: bool(d.get(k)) for k in READINESS_ITEMS}
    out.update({"email": email, "cradle_source": d.get("cradle_source")})
    out["complete"] = readiness_complete(out)
    return out


def set_readiness_item(cx, email: str, item: str, value: bool,
                       *, cradle_source: str | None = None) -> dict:
    email = (email or "").strip().lower()
    if item not in READINESS_ITEMS:
        raise ValueError(f"unknown readiness item: {item}")
    now = datetime.utcnow().isoformat()
    cx.execute("INSERT OR IGNORE INTO evox_readiness (email, updated_at) VALUES (?,?)",
               (email, now))
    cx.execute(f"UPDATE evox_readiness SET {item}=?, updated_at=? WHERE email=?",
               (1 if value else 0, now, email))
    if item == "cradle_ok" and cradle_source is not None:
        cx.execute("UPDATE evox_readiness SET cradle_source=? WHERE email=?",
                   (cradle_source, email))
    state = get_readiness(cx, email)
    if state["complete"]:
        cx.execute("UPDATE evox_readiness SET completed_at=COALESCE(completed_at,?) "
                   "WHERE email=?", (now, email))
    cx.commit()
    return get_readiness(cx, email)
