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
# 60 chars comfortably covers a real display name ("Dr. Jane Q. Doe-Smith, RN")
# while remaining far too short for the HTML/URL/phone/email payloads sanitize()
# strips -- matching how MAX_BODY bounds `body`.
MAX_DISPLAY_NAME = 60

# Only strips things that actually look like tags: '<' (or '</'), with NO
# whitespace before the tag-name letter, then an ASCII letter (real tags and
# closing tags are always written this tight — '<script', '</div'). A bare
# comparison operator is almost always written with a space after the '<'
# ("A < B", "5 < 10"), so requiring the letter immediately adjacent is enough
# to tell the two apart without a false split on either side.
_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")

# Scheme-prefixed and www-prefixed URLs ONLY. We deliberately do NOT strip
# bare domains typed without a scheme (e.g. "evil.com"): a missing space
# after a sentence-ending period followed by a capitalized word that happens
# to match a TLD in the allowlist ("years.Health", "up.Co") swallows both
# words — "Health" and "Clinic" are exactly what clients write on a health
# site. A bare domain is inert under .textContent rendering and the human
# approval gate downstream catches it; a scheme or "www." prefix is required
# before we treat it as an active link worth removing.
_URL_RE = re.compile(r"\b(?:https?://\S+|www\.\S+)", re.I)

# javascript:/data:/vbscript: URI schemes. Not exploitable under textContent
# rendering today, but sanitize() is general-purpose and these are the classic
# href-injection vectors if it's ever reused in an href-bearing context.
_SCHEME_RE = re.compile(r"\b(?:javascript|data|vbscript):\S*", re.I)

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

# Phone numbers: only digit groups joined by an actual separator (space,
# dash, dot, parens), or anything led by a '+' (international format, which
# may or may not use separators). We deliberately do NOT strip a bare,
# unspaced run of digits: "my order 1234567 arrived", "tracking number
# 1234567890 shipped", and "batch made on 20260315" are all legitimate
# 280-char client prose, not phone numbers, and a bare digit run is inert
# under .textContent — the human approval gate catches it if it really is a
# phone number typed without separators.
_PHONE_RE = re.compile(
    r"\+\d[\d\-\.\(\) ]*\d"
    r"|\b\d{1,4}(?:[\-\.\(\) ]+\d{1,4}){1,}\b"
)


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
    out = _SCHEME_RE.sub("", out)
    # Email before URL: an address like a@sub.example.co.uk contains a bare
    # domain that _URL_RE would otherwise match and remove first, stranding
    # the "a@" local part instead of stripping the whole address.
    out = _EMAIL_RE.sub("", out)
    out = _URL_RE.sub("", out)
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
    # display_name gets the same treatment as body: sanitize() strips HTML/URLs/
    # emails/phone numbers, then a length cap. The <input maxlength="40"> in
    # client-portal.html is client-side only and enforces nothing server-side.
    clean_name = sanitize(display_name)
    if len(clean_name) > MAX_DISPLAY_NAME:
        raise ValueError(f"display_name exceeds {MAX_DISPLAY_NAME} characters after sanitizing")
    if not clean_name:
        raise ValueError("display_name is empty after sanitizing")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cx.execute("""
        INSERT INTO share_headers (email, display_name, body, status, created_at, updated_at)
        VALUES (?,?,?, 'pending', ?, ?)
        ON CONFLICT(email) DO UPDATE SET
            display_name=excluded.display_name,
            body=excluded.body,
            status='pending',
            updated_at=excluded.updated_at
    """, ((email or "").strip().lower(), clean_name, clean, now, now))
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


def list_pending(cx):
    """All rows still awaiting review, oldest first. Without this the
    approval queue is invisible -- headers get approved blind (by email,
    guessed) or never reviewed at all."""
    rows = cx.execute(
        "SELECT email, display_name, body, created_at FROM share_headers"
        " WHERE status='pending' ORDER BY created_at ASC").fetchall()
    return [dict(r) for r in rows]
