"""EVOX booking: self-attest readiness, availability, 1:1 phone booking, ICS.
Pure helpers take primitives only (no cx) and must import without importing app."""
import sqlite3
import secrets
import json as _json
from datetime import datetime, timedelta, timezone

READINESS_ITEMS = ("pc_ok", "cradle_ok", "headset_ok", "zyto_ok")


class SlotTaken(Exception):
    pass


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
    for _col, _decl in (("session_type", "TEXT DEFAULT 'evox'"),
                        ("medium", "TEXT DEFAULT 'phone'"),
                        ("zoom_join_url", "TEXT"), ("zoom_meeting_id", "TEXT")):
        try:
            cx.execute(f"ALTER TABLE evox_bookings ADD COLUMN {_col} {_decl}")
        except Exception:
            pass
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


def booked_starts(cx, practitioner: str = "rae") -> set:
    rows = cx.execute("SELECT start_ts FROM evox_bookings "
                      "WHERE practitioner=? AND status='booked'", (practitioner,)).fetchall()
    return {r[0] for r in rows}


def rae_busy_intervals(cx, lo_date: str, hi_date: str, practitioner: str = "rae"):
    rows = cx.execute(
        "SELECT start, end FROM calendar_events WHERE owner=? AND status='visible' "
        "AND substr(start,1,10) BETWEEN ? AND ?", (practitioner, lo_date, hi_date)).fetchall()
    return [(r[0], r[1] or "") for r in rows]


def create_booking(cx, email: str, start_ts: str, *, duration_min: int = 60,
                   prepaid: bool = False, practitioner: str = "rae",
                   session_type: str = "evox", medium: str = "phone", tag_fn=None) -> dict:
    email = (email or "").strip().lower()
    start_dt = datetime.fromisoformat(start_ts[:19])
    end_ts = (start_dt + timedelta(minutes=duration_min)).isoformat()
    now = datetime.now(timezone.utc).isoformat()
    ics_uid = f"{session_type}-{secrets.token_hex(8)}@illtowell.com"
    try:
        cur = cx.execute(
            "INSERT INTO evox_bookings (email,practitioner,start_ts,end_ts,status,"
            "prepaid,ics_uid,created_at,session_type,medium) "
            "VALUES (?,?,?,?,'booked',?,?,?,?,?)",
            (email, practitioner, start_ts, end_ts, 1 if prepaid else 0, ics_uid, now,
             session_type, medium))
    except sqlite3.IntegrityError as e:
        cx.rollback()
        if "UNIQUE" in str(e).upper():
            raise SlotTaken(start_ts)
        raise
    booking_id = cur.lastrowid
    ev_id = f"{session_type}-{booking_id}"
    location = "Video" if medium == "video" else "Phone"
    label = "Biofield Consult" if session_type == "biofield-consult" else "EVOX"
    cx.execute(
        "INSERT INTO calendar_events (pushed_at,google_cal_id,google_event_id,"
        "calendar_name,summary,start,end,location,owner,status,cal_alert) "
        "VALUES (?, 'delegated', ?, ?, ?, ?, ?, ?, ?, 'visible', 0)",
        (now, ev_id, f"{label} booking", f"{label} — {email}", start_ts, end_ts,
         location, practitioner))
    cx.execute("UPDATE evox_bookings SET calendar_event_id=? WHERE id=?", (ev_id, booking_id))
    cx.commit()
    if tag_fn:
        tag_fn(email, ["evox-client", "evox-ready"])
    return {"id": booking_id, "email": email, "start_ts": start_ts, "end_ts": end_ts,
            "ics_uid": ics_uid, "prepaid": prepaid, "session_type": session_type,
            "medium": medium}


def build_ics(*, uid, start_ts, end_ts, summary, description, location,
              organizer_email="rae@illtowell.com") -> bytes:
    def _fmt(ts):  # naive local -> floating VEVENT time
        return datetime.fromisoformat(ts[:19]).strftime("%Y%m%dT%H%M%S")
    dtstamp = datetime.fromisoformat(start_ts[:19]).strftime("%Y%m%dT000000")
    desc = (description or "").replace("\n", "\\n")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//illtowell//EVOX//EN",
             "METHOD:REQUEST", "BEGIN:VEVENT", f"UID:{uid}", f"DTSTAMP:{dtstamp}",
             f"DTSTART:{_fmt(start_ts)}", f"DTEND:{_fmt(end_ts)}",
             f"SUMMARY:{summary}", f"DESCRIPTION:{desc}", f"LOCATION:{location}",
             f"ORGANIZER:mailto:{organizer_email}", "STATUS:CONFIRMED",
             "END:VEVENT", "END:VCALENDAR"]
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def session_credit_balance(cx, email: str) -> int:
    email = (email or "").strip().lower()
    r = cx.execute("SELECT credits FROM evox_session_credits WHERE email=?", (email,)).fetchone()
    return int(r[0]) if r else 0


def add_session_credits(cx, email: str, n: int) -> int:
    email = (email or "").strip().lower()
    cx.execute("INSERT INTO evox_session_credits (email, credits) VALUES (?, ?) "
               "ON CONFLICT(email) DO UPDATE SET credits=credits+excluded.credits", (email, n))
    cx.commit()
    return session_credit_balance(cx, email)


def consume_session_credit(cx, email: str) -> bool:
    email = (email or "").strip().lower()
    cur = cx.execute("UPDATE evox_session_credits SET credits=credits-1 "
                     "WHERE email=? AND credits>0", (email,))
    cx.commit()
    return cur.rowcount > 0


def has_cradle_purchase(cx, email: str) -> bool:
    email = (email or "").strip().lower()
    try:
        rows = cx.execute("SELECT items_json FROM orders WHERE lower(email)=?", (email,)).fetchall()
    except sqlite3.OperationalError:
        return False
    for (items,) in rows:
        try:
            for line in _json.loads(items or "[]"):
                if (line.get("slug") or "") == "hand-cradle":
                    return True
        except Exception:
            continue
    return False


def ensure_portal_token(cx, email, name):
    """Return a usable raw portal token for `email` — creating the portal on first
    touch — that round-trips through portal_identity.resolve_identity. Delegates to
    client_portal.ensure_token, which holds a STABLE raw token in portal_notify_state
    (only its one-way hash lives on the portal row), so calling this repeatedly for
    the same email yields the SAME non-empty token rather than rotating the link."""
    from dashboard import client_portal as _cp
    return _cp.ensure_token(cx, email, name)
