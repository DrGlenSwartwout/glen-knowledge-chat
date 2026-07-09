"""Triage/Discovery invites: tokenized, hashed, single-use, 7-day expiry.
Stdlib-only; import without importing app."""
import sqlite3, secrets, hashlib
from datetime import datetime, timezone, timedelta

from dashboard.timeutil import parse_utc as _parse_utc

def _hash(token: str) -> str:
    return hashlib.sha256((token or "").encode()).hexdigest()

def init_triage_tables(cx) -> None:
    cx.execute("""CREATE TABLE IF NOT EXISTS triage_invites (
        token_hash TEXT PRIMARY KEY, email TEXT, name TEXT, practitioner TEXT,
        status TEXT DEFAULT 'invited', created_at TEXT, expires_at TEXT,
        booked_start TEXT)""")
    cx.commit()

def create_invite(cx, email, name, practitioner, *, days: int = 7, _now=None) -> str:
    now = _now or datetime.now(timezone.utc)
    email = (email or "").strip().lower()
    token = secrets.token_urlsafe(24)
    cx.execute("INSERT INTO triage_invites (token_hash, email, name, practitioner, "
               "status, created_at, expires_at) VALUES (?,?,?,?, 'invited', ?, ?)",
               (_hash(token), email, (name or "").strip(), practitioner,
                now.isoformat(), (now + timedelta(days=days)).isoformat()))
    cx.commit()
    return token

def resolve_invite(cx, token, *, _now=None):
    now = _now or datetime.now(timezone.utc)
    cur = cx.execute("SELECT email,name,practitioner,status,created_at,expires_at,booked_start "
                     "FROM triage_invites WHERE token_hash=?", (_hash(token),))
    cols = [c[0] for c in cur.description]; r = cur.fetchone()
    if r is None:
        return None
    d = dict(zip(cols, r))
    if d.get("status") == "cancelled":
        return None
    try:
        expired = _parse_utc(d["expires_at"]) < _parse_utc(now)
    except (ValueError, TypeError):
        # Unchanged from before: a row with a missing or garbled expiry has
        # never counted as expired here. Only the parsing got more tolerant.
        expired = False
    if expired:
        return None
    return {"email": d["email"], "name": d["name"], "practitioner": d["practitioner"],
            "status": d["status"], "booked_start": d.get("booked_start")}

def mark_booked(cx, token, start_ts) -> None:
    cx.execute("UPDATE triage_invites SET status='booked', booked_start=? WHERE token_hash=?",
               (start_ts, _hash(token)))
    cx.commit()
