"""Per-client notification preference + engagement state (Phase 1 of scan-notify).
One row per email: opt status (default/in/out), notify_count (taper), engaged,
phone, and the stored stable portal token for the notification link."""
import datetime
import sqlite3

MAX_TAPER = 3  # default (non-engaged, non-opted) clients get this many, then quiet


def _now():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def init_table(cx):
    cx.execute("""CREATE TABLE IF NOT EXISTS portal_notify_state (
        email TEXT PRIMARY KEY, phone TEXT, opt_status TEXT DEFAULT 'default',
        notify_count INTEGER DEFAULT 0, engaged INTEGER DEFAULT 0,
        portal_token TEXT, updated_at TEXT)""")
    cx.commit()


def _norm(email):
    return (email or "").strip().lower()


def get_state(cx, email):
    init_table(cx)
    row = cx.execute("SELECT email, phone, opt_status, notify_count, engaged, portal_token "
                     "FROM portal_notify_state WHERE email=?", (_norm(email),)).fetchone()
    if not row:
        return {"email": _norm(email), "phone": "", "opt_status": "default",
                "notify_count": 0, "engaged": False, "portal_token": ""}
    return {"email": row[0], "phone": row[1] or "", "opt_status": row[2] or "default",
            "notify_count": row[3] or 0, "engaged": bool(row[4]), "portal_token": row[5] or ""}


def _upsert(cx, email, **fields):
    init_table(cx)
    email = _norm(email)
    if not cx.execute("SELECT 1 FROM portal_notify_state WHERE email=?", (email,)).fetchone():
        cx.execute("INSERT INTO portal_notify_state (email, updated_at) VALUES (?,?)", (email, _now()))
    sets = ", ".join(f"{k}=?" for k in fields) + ", updated_at=?"
    cx.execute(f"UPDATE portal_notify_state SET {sets} WHERE email=?",
               (*fields.values(), _now(), email))
    cx.commit()


def set_opt(cx, email, status):
    _upsert(cx, email, opt_status=status)


def set_phone(cx, email, phone):
    _upsert(cx, email, phone=(phone or "").strip())


def set_token(cx, email, token):
    _upsert(cx, email, portal_token=token or "")


def mark_engaged(cx, email):
    _upsert(cx, email, engaged=1)


def incr_notify(cx, email):
    s = get_state(cx, email)
    _upsert(cx, email, notify_count=s["notify_count"] + 1)


def decide(state):
    """Eligibility + which of the 3 message variants to send (0,1,2) or None."""
    if state["opt_status"] == "out":
        return {"eligible": False, "variant": None}
    if state["opt_status"] == "in" or state["engaged"]:
        return {"eligible": True, "variant": min(state["notify_count"], MAX_TAPER - 1)}
    if state["notify_count"] < MAX_TAPER:
        return {"eligible": True, "variant": state["notify_count"]}
    return {"eligible": False, "variant": None}


def email_by_phone(cx, phone):
    """Reverse lookup for the Twilio inbound (STOP/START) webhook, by last-10 digits."""
    init_table(cx)
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())[-10:]
    if not digits:
        return None
    for row in cx.execute("SELECT email, phone FROM portal_notify_state").fetchall():
        if "".join(ch for ch in (row[1] or "") if ch.isdigit()).endswith(digits):
            return row[0]
    return None
