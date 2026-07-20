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

# Only strips things that actually look like tags: '<', optional whitespace,
# optional '/', then an ASCII letter (real tags/closing-tags always start
# this way). A bare comparison operator like "5 < 10 > 3" never has a letter
# right after the '<', so it survives untouched.
_TAG_RE = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")

# Scheme/www-prefixed URLs, PLUS bare domains ("evil.com", "evil.com/path")
# under a common TLD allowlist. A bare-domain match requires a real TLD so we
# don't eat ordinary words with dots (there aren't many in a 280-char message,
# but this keeps the match intentional rather than "anything.anything").
_URL_RE = re.compile(
    r"\b(?:https?://|www\.)\S+"
    r"|\b[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*"
    r"\.(?:com|net|org|io|co|health|clinic)\b(?:/\S*)?",
    re.I,
)

# javascript:/data:/vbscript: URI schemes. Not exploitable under textContent
# rendering today, but sanitize() is general-purpose and these are the classic
# href-injection vectors if it's ever reused in an href-bearing context.
_SCHEME_RE = re.compile(r"\b(?:javascript|data|vbscript):\S*", re.I)

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

# Phone numbers: the first alternative below (\+?\d[\d\-\.\(\)\ ]{6,}\d)
# already catches any separated form (808-555-1212, (808) 555-1212) AND any
# unspaced run of 8+ digits, because its middle character class includes
# plain digits alongside separators — so a bare 10-digit run is already
# covered by that clause. The one real gap is the unspaced 7-digit run (a
# US local number with no area code, e.g. "5551212") — one digit too short
# to hit the {6,} minimum. The second alternative below plugs exactly that
# gap; the third (10-digit) is kept explicit for readability/defense-in-depth
# even though it's redundant with the first. We deliberately do NOT add a
# bare 1-6 digit unspaced pattern: a bare "2026" year, a "42" price, or
# "6 months" are legitimate content a client would write in a 280-char
# message, and matching short digit runs would eat them.
_PHONE_RE = re.compile(
    r"\b(?:\+?\d[\d\-\.\(\) ]{6,}\d)\b"
    r"|\+?\b\d{7}\b"
    r"|\+?\b\d{10}\b"
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
