"""Client-authored header shown above the public sample portal.

Why an approval gate: self-authored is not the same as risk-free once it is
published on a commercial domain beside a product grid. CaringBridge is a
neutral platform where patients write about themselves; a client's sentence on
illtowell.com is a TESTIMONIAL, and FTC endorsement rules attach. "This cleared
my glaucoma" would be a health claim Glen published. Headers therefore land as
'pending' and render only once approved.
"""

import re
from datetime import datetime, timezone

MAX_BODY = 280

_TAG_RE = re.compile(r"<[^>]*>")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d\-\.\(\) ]{6,}\d)\b")


def init_share_headers_table(cx):
    cx.execute("""
        CREATE TABLE IF NOT EXISTS share_headers (
            email        TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            body         TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        )
    """)
    cx.execute("CREATE INDEX IF NOT EXISTS ix_sh_status ON share_headers(status)")
    cx.commit()


def sanitize(body):
    """Strip HTML, URLs, emails and phone numbers. Collapse whitespace."""
    out = _TAG_RE.sub("", body or "")
    out = _URL_RE.sub("", out)
    out = _EMAIL_RE.sub("", out)
    out = _PHONE_RE.sub("", out)
    return " ".join(out.split()).strip()


def upsert_header(cx, email, display_name, body):
    """Write a header for `email`, always as pending. Editing an approved
    header resets it to pending — re-review is the point."""
    clean = sanitize(body)
    if len(clean) > MAX_BODY:
        raise ValueError(f"body exceeds {MAX_BODY} characters after sanitizing")
    if not clean:
        raise ValueError("body is empty after sanitizing")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cx.execute("""
        INSERT INTO share_headers (email, display_name, body, status, created_at, updated_at)
        VALUES (?,?,?, 'pending', ?, ?)
        ON CONFLICT(email) DO UPDATE SET
            display_name=excluded.display_name,
            body=excluded.body,
            status='pending',
            updated_at=excluded.updated_at
    """, ((email or "").strip().lower(), (display_name or "").strip(), clean, now, now))
    cx.commit()
    return dict(cx.execute("SELECT * FROM share_headers WHERE email=?",
                           ((email or "").strip().lower(),)).fetchone())


def _set_status(cx, email, status):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cx.execute("UPDATE share_headers SET status=?, updated_at=? WHERE email=?",
               (status, now, (email or "").strip().lower()))
    cx.commit()


def approve(cx, email):
    _set_status(cx, email, "approved")


def reject(cx, email):
    _set_status(cx, email, "rejected")


def get_approved(cx, email):
    row = cx.execute(
        "SELECT display_name, body FROM share_headers WHERE email=? AND status='approved'",
        ((email or "").strip().lower(),)).fetchone()
    return dict(row) if row else None
