"""MentorshipU intake: scoped access-link sessions.

Modeled on dashboard/intake_public.py's scoped-token pattern. A MentorshipU
access-link session is a random, expiring token bound to the email/name the
setup link was minted for. Pure logic only: no Flask, no network, stdlib only.
"""
from __future__ import annotations

import secrets
import time

TOKEN_TTL_HOURS = 72


def init_mentorship_sessions_table(cx) -> None:
    cx.execute(
        "CREATE TABLE IF NOT EXISTS mentorship_sessions("
        "token TEXT PRIMARY KEY, email TEXT, name TEXT, "
        "created_at TEXT, expires_at REAL)"
    )
    cx.commit()


def create_session(cx, email: str, name: str, now: float | None = None) -> str:
    now = time.time() if now is None else now
    token = secrets.token_urlsafe(32)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    cx.execute(
        "INSERT INTO mentorship_sessions(token, email, name, created_at, expires_at) "
        "VALUES(?,?,?,?,?)",
        (token, email, name, created, now + TOKEN_TTL_HOURS * 3600),
    )
    cx.commit()
    return token


def resolve_session(cx, token: str, now: float | None = None) -> str | None:
    now = time.time() if now is None else now
    row = cx.execute(
        "SELECT email, expires_at FROM mentorship_sessions WHERE token=?", (token,)
    ).fetchone()
    if not row:
        return None
    email, exp = row[0], row[1]
    if exp is not None and now > float(exp):
        return None
    return email
