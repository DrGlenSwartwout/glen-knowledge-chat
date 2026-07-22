"""Durable, opaque, per-recipient tokens for email/newsletter tracked links.

A tracked email link carries a random token (never the recipient's email), so
the redirect can resolve identity server-side with NO PII in the URL. This is a
DEDICATED token type — deliberately NOT the client portal token — so a forwarded
email link can never grant portal access.
"""
import secrets
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _norm(email):
    return (email or "").strip().lower()


def init_email_click_tokens(cx):
    cx.execute(
        "CREATE TABLE IF NOT EXISTS email_click_tokens ("
        "token TEXT PRIMARY KEY, email TEXT NOT NULL, created_at TEXT)"
    )
    cx.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_email_click_tokens_email "
        "ON email_click_tokens(email)"
    )
    cx.commit()


def token_for(cx, email):
    """Return the recipient's durable token, minting once and reusing after."""
    e = _norm(email)
    if not e:
        return ""
    row = cx.execute(
        "SELECT token FROM email_click_tokens WHERE email=?", (e,)
    ).fetchone()
    if row and row[0]:
        return row[0]
    token = secrets.token_urlsafe(24)
    cx.execute(
        "INSERT OR IGNORE INTO email_click_tokens (token, email, created_at) "
        "VALUES (?, ?, ?)", (token, e, _now())
    )
    cx.commit()
    # Re-read in case of a concurrent insert winning the unique(email) race.
    row = cx.execute(
        "SELECT token FROM email_click_tokens WHERE email=?", (e,)
    ).fetchone()
    return row[0] if row else token


def email_for(cx, token):
    """Resolve a token to its normalized email, or None."""
    t = (token or "").strip()
    if not t:
        return None
    row = cx.execute(
        "SELECT email FROM email_click_tokens WHERE token=?", (t,)
    ).fetchone()
    return _norm(row[0]) if row and row[0] else None
