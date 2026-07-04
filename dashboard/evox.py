"""EVOX booking: self-attest readiness, availability, 1:1 phone booking, ICS.
Pure helpers take primitives only (no cx) and must import without importing app."""
import sqlite3
from datetime import datetime, timedelta, timezone

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
    cur = cx.execute("SELECT * FROM evox_readiness WHERE email=?", (email,))
    cols = [c[0] for c in cur.description]
    r = cur.fetchone()
    row = dict(zip(cols, r)) if r is not None else None
    if row is None:
        base = {k: False for k in READINESS_ITEMS}
        base.update({"email": email, "cradle_source": None, "complete": False})
        return base
    out = {k: bool(row.get(k)) for k in READINESS_ITEMS}
    out.update({"email": email, "cradle_source": row.get("cradle_source")})
    out["complete"] = readiness_complete(out)
    return out


def set_readiness_item(cx, email: str, item: str, value: bool,
                       *, cradle_source: str | None = None) -> dict:
    email = (email or "").strip().lower()
    if item not in READINESS_ITEMS:
        raise ValueError(f"unknown readiness item: {item}")
    now = datetime.now(timezone.utc).isoformat()
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


# Pure availability computation helpers (no DB, no app import)

def parse_office_hours(spec: str):
    days_part, hours_part = spec.split(":", 1)
    lo, hi = days_part.split("-")
    start_hm, end_hm = hours_part.split("-")
    return int(lo), int(hi), start_hm, end_hm


def _hm(day, hm: str) -> datetime:
    h, m = hm.split(":")
    return datetime(day.year, day.month, day.day, int(h), int(m))


def slot_grid(day, spec: str, duration_min: int = 60):
    lo, hi, start_hm, end_hm = parse_office_hours(spec)
    if not (lo <= day.isoweekday() <= hi):
        return []
    start, end = _hm(day, start_hm), _hm(day, end_hm)
    out, t, step = [], start, timedelta(minutes=duration_min)
    while t + step <= end:
        out.append(t.isoformat()); t += step
    return out


def _parse(ts: str):
    try:
        ts = (ts or "").strip()
        if not ts:
            return None
        if len(ts) == 10:            # date-only, e.g. all-day event
            return datetime.fromisoformat(ts + "T00:00:00")
        return datetime.fromisoformat(ts[:19])
    except (ValueError, AttributeError, TypeError):
        return None


def intervals_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


def available_slots(days, office_spec, busy, booked, now, duration_min: int = 60):
    step = timedelta(minutes=duration_min)
    # Normalize busy into datetime intervals; date-only start w/ empty end = whole day.
    intervals = []
    for bs, be in busy:
        s = _parse(bs)
        if s is None:
            continue
        if len(str(bs).strip()) == 10 and not (be or "").strip():
            e = s + timedelta(days=1)
        else:
            e = _parse(be) or (s + step)
        intervals.append((s, e))
    out = []
    for day in days:
        for iso in slot_grid(day, office_spec, duration_min):
            s = datetime.fromisoformat(iso)
            if s <= now or iso in booked:
                continue
            e = s + step
            if any(intervals_overlap(s, e, bs, be) for bs, be in intervals):
                continue
            out.append(iso)
    return sorted(out)
