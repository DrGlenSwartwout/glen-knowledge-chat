"""Portal welcome email — once-per-member send guard + content builder.

Step 3 (#342) guaranteed every member a `people` row so self-login works.
This module tells a new member, once, that their portal is ready.

Pure / stdlib-only so it is offline-testable. The actual network send and the
join-flow hooks live in app.py (which has the GHL/SMTP + suppression context).
"""

from datetime import datetime


def mark_welcome_sent(cx, email):
    """Once-per-email guard. Ensures the `portal_welcome_sent` table exists,
    then INSERT OR IGNOREs the lowercased email.

    Returns True iff a row was newly inserted (first time → caller should send),
    False if the email was already present (already sent → skip). Idempotent;
    case-insensitive.
    """
    em = (email or "").strip().lower()
    if not em:
        return False
    cx.execute(
        "CREATE TABLE IF NOT EXISTS portal_welcome_sent ("
        "  email TEXT PRIMARY KEY,"
        "  sent_at TEXT"
        ")"
    )
    cur = cx.execute(
        "INSERT OR IGNORE INTO portal_welcome_sent (email, sent_at) VALUES (?, ?)",
        (em, datetime.utcnow().isoformat() + "Z"),
    )
    return cur.rowcount > 0


def _first_name(name):
    n = (name or "").strip()
    if not n:
        return ""
    first = n.split()[0]
    # Guard against an email-as-name slipping through (e.g. "this.elf@gmail.com").
    if "@" in first or "." in first:
        return ""
    return first


def welcome_email_content(name, login_url):
    """Pure builder → (subject, text_body, html_body). Glen's voice; greets by
    first name when known, falls back to "there". Points to login_url
    (self-serve /portal/login). No token, no secret.
    """
    first = _first_name(name)
    greeting = f"Aloha {first}," if first else "Aloha,"
    subject = "Your Healing Oasis portal is ready"

    text_body = (
        f"{greeting}\n\n"
        "Your personal Healing Oasis portal is ready. It's your private home for "
        "everything we do together — your Biofield Analysis reports, your remedy "
        "schedule, easy reordering, and your Ambassador links if you choose to share.\n\n"
        "Sign in anytime with just your email — no password to remember:\n\n"
        f"{login_url}\n\n"
        "Enter your email there and we'll send you a secure one-time link.\n\n"
        "With aloha,\n"
        "Dr. Glen & Rae\n"
        "Remedy Match LLC\n"
    )

    html_body = (
        f"<p>{greeting}</p>"
        "<p>Your personal Healing Oasis portal is ready. It's your private home for "
        "everything we do together — your Biofield Analysis reports, your remedy "
        "schedule, easy reordering, and your Ambassador links if you choose to share.</p>"
        "<p>Sign in anytime with just your email — no password to remember:</p>"
        f"<p><a href=\"{login_url}\">Open your portal</a></p>"
        f"<p style=\"color:#666;font-size:12px;\">Or paste this URL into your browser: {login_url}<br>"
        "Enter your email there and we'll send you a secure one-time link.</p>"
        "<p>With aloha,<br><b>Dr. Glen &amp; Rae</b><br>Remedy Match LLC</p>"
    )
    return subject, text_body, html_body
