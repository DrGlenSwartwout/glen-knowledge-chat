"""Inbox — Gmail thread list, full-message read, and reply send for /console/inbox.

Reuses the same OAuth token loading pattern as reply_watcher.py. The token
is written by ~/AI-Training/02 Skills/google-auth.py locally and lives at
/data/google-token.json on Render (set via GMAIL_TOKEN_PATH env var).

Scope `gmail.modify` already covers read + modify + send.

Public surface used by app.py routes:
    list_threads(query, max_results)         → [{id, subject, sender, snippet, date, labels, unread}, ...]
    get_thread(thread_id)                    → {id, subject, messages: [...]} with decoded bodies
    send_reply(thread_id, body, to=None)     → posts a reply to the most recent message
    archive_thread(thread_id)                → removes INBOX label
    star_thread(thread_id) / unstar_thread   → toggles STARRED
    mark_read(thread_id) / mark_unread       → toggles UNREAD
"""

from __future__ import annotations

import base64
import os
import re
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional


# Granted scopes the local OAuth flow asks for. Operations needing label
# modification (archive/star/mark-unread) will require gmail.modify added
# to google-auth.py and a re-auth — the current token doesn't include it,
# so those API calls will return 403 until Glen re-authorizes.
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
]

# Token-path resolution: env var override → Render persistent disk →
# local home-dir convention. First file that exists wins.
_TOKEN_PATH_CANDIDATES = [
    "/data/google-token.json",                                     # Render persistent disk
    str(Path.home() / ".config" / "google" / "token.json"),        # local dev
]


def _resolve_token_path() -> Path:
    """Return the first existing token path among env override + known locations."""
    env_override = os.environ.get("GMAIL_TOKEN_PATH")
    candidates = ([env_override] if env_override else []) + _TOKEN_PATH_CANDIDATES
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise RuntimeError(
        f"No Gmail token at any of: {[c for c in candidates if c]}. "
        f"Run '~/AI-Training/02 Skills/google-auth.py' locally then POST it to "
        f"/admin/upload-gmail-token to land it on the Render disk."
    )


def _get_gmail_service():
    """Build a Gmail API service client.

    Critical: passes the intersection of requested scopes and what the token
    was actually granted. If we ask for a scope the token doesn't have (e.g.,
    gmail.modify when the token was issued for read+send only), the SDK
    requests it on token refresh and Google returns `invalid_scope: Bad
    Request`. Reading the file's `scopes` field and intersecting avoids that.
    """
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    import json as _json

    token_path = _resolve_token_path()
    with open(token_path) as f:
        granted = set((_json.load(f) or {}).get("scopes") or [])
    # Use granted scopes only (intersection with what we'd want), else all granted
    effective = list(set(GMAIL_SCOPES) & granted) if granted else list(GMAIL_SCOPES)
    if not effective:
        effective = list(granted) or list(GMAIL_SCOPES)
    creds = Credentials.from_authorized_user_file(str(token_path), scopes=effective)
    return build("gmail", "v1", credentials=creds)


# ── Decoding helpers (pure — testable) ────────────────────────────────────────

def _decode_b64url(data: str) -> str:
    """Gmail uses URL-safe base64 without padding. Add padding back and decode."""
    if not data:
        return ""
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad).decode("utf-8", errors="replace")


def _strip_html_to_text(html: str) -> str:
    """Convert HTML to readable plain text. Strips <style>/<script> blocks
    INCLUDING their contents (so the CSS doesn't leak), then strips remaining
    tags, then decodes HTML entities (numeric + named) properly via stdlib."""
    if not html:
        return ""
    import html as _html_mod
    # Order matters: drop style/script CONTENTS first, before tag-stripping
    s = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.S | re.I)
    s = re.sub(r"<script[^>]*>.*?</script>", "", s, flags=re.S | re.I)
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    # Convert structural tags to whitespace before nuking the rest
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p\s*>|</div\s*>|</tr\s*>|</li\s*>|</h[1-6]\s*>", "\n", s, flags=re.I)
    # Strip remaining tags
    s = re.sub(r"<[^>]+>", "", s)
    # Decode entities (handles &#039;, &#064;, &amp;, &nbsp;, etc.)
    s = _html_mod.unescape(s)
    # Collapse runs of whitespace inside lines, preserve paragraph breaks
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_body(payload: dict) -> dict:
    """Walk a Gmail message payload tree and return {plain, html} bodies.

    Prefers text/plain when present. Otherwise renders HTML to readable plain
    text via _strip_html_to_text (drops <style>/<script> contents, decodes
    entities, normalizes whitespace).
    """
    plain = ""
    html = ""

    def walk(p: dict):
        nonlocal plain, html
        mime = (p.get("mimeType") or "").lower()
        body = p.get("body") or {}
        data = body.get("data")
        if mime == "text/plain" and data and not plain:
            plain = _decode_b64url(data)
        elif mime == "text/html" and data and not html:
            html = _decode_b64url(data)
        for part in (p.get("parts") or []):
            walk(part)

    walk(payload or {})
    if not plain and html:
        plain = _strip_html_to_text(html)
    return {"plain": plain, "html": html}


def _header(headers: list, name: str) -> str:
    name_l = name.lower()
    for h in headers or []:
        if h.get("name", "").lower() == name_l:
            return h.get("value", "")
    return ""


def categorize(labels: list) -> str:
    """Bucket a thread into a coarse category for filtering.

    Returns one of: 'promotions', 'social', 'updates', 'forums',
    'important', 'inbox' (everything else with INBOX label).
    Gmail's CATEGORY_* labels carry most of the signal; IMPORTANT
    overrides categories so a flagged "promotion" still surfaces.
    """
    s = set(labels or [])
    if "IMPORTANT" in s or "STARRED" in s:
        return "important"
    if "CATEGORY_PROMOTIONS" in s:
        return "promotions"
    if "CATEGORY_SOCIAL" in s:
        return "social"
    if "CATEGORY_UPDATES" in s:
        return "updates"
    if "CATEGORY_FORUMS" in s:
        return "forums"
    return "inbox"


# ── Body cleaning ────────────────────────────────────────────────────────────
# Strip HTML, drop quoted-reply chains, trim email signatures so the UI
# shows the actual message content rather than a wall of forwarded chrome.

import re as _re

_QUOTE_HEADER_PATTERNS = [
    # "On Mon, Apr 28, 2026 at 12:11 PM Practice Better <...> wrote:"
    _re.compile(r"^\s*On\s+\w+,?\s+\w+\s+\d+,?\s+\d{4}.*?wrote:\s*$", _re.M),
    # "On 4/28/26 at 12:11 PM, ... wrote:"
    _re.compile(r"^\s*On\s+\d+/\d+/\d+.*?wrote:\s*$", _re.M),
    # "From: ...\nSent: ...\nTo: ...\nSubject: ..." (Outlook-style header block)
    _re.compile(r"^\s*From:\s+.+\n(\s*Sent:\s+.+\n)?\s*To:\s+.+", _re.M),
    # Gmail forward marker
    _re.compile(r"^-{2,}\s*Forwarded message\s*-{2,}\s*$", _re.M | _re.I),
    # Standard reply marker
    _re.compile(r"^-{2,}\s*Original Message\s*-{2,}\s*$", _re.M | _re.I),
]

_SIGNATURE_PATTERNS = [
    _re.compile(r"^\s*--\s*$", _re.M),  # Standard --\n signature delimiter
    _re.compile(r"^\s*Sent from my (iPhone|iPad|Android|mobile).*$", _re.M | _re.I),
    _re.compile(r"^\s*Get Outlook for (iOS|Android).*$", _re.M | _re.I),
]


def clean_body(text: str) -> str:
    """Strip HTML tags, quoted-reply chains, and trailing signatures.

    Heuristic-based. Optimized for email content where preserving the
    'core message' matters more than perfect structure.
    """
    if not text:
        return ""
    # Strip HTML if any leaked through
    s = _re.sub(r"<style[^>]*>.*?</style>", "", text, flags=_re.S | _re.I)
    s = _re.sub(r"<script[^>]*>.*?</script>", "", s, flags=_re.S | _re.I)
    s = _re.sub(r"<br\s*/?>", "\n", s, flags=_re.I)
    s = _re.sub(r"</p>", "\n\n", s, flags=_re.I)
    s = _re.sub(r"<[^>]+>", "", s)
    # Decode common HTML entities
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
           .replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", '"'))
    # Cut at first quoted-history marker
    earliest = len(s)
    for pat in _QUOTE_HEADER_PATTERNS:
        m = pat.search(s)
        if m and m.start() < earliest:
            earliest = m.start()
    s = s[:earliest]
    # Cut at signature delimiter
    for pat in _SIGNATURE_PATTERNS:
        m = pat.search(s)
        if m:
            s = s[:m.start()]
    # Strip "> quoted line" prefixes
    s = _re.sub(r"^>\s.*$", "", s, flags=_re.M)
    # Collapse runs of blank lines
    s = _re.sub(r"\n{3,}", "\n\n", s)
    # Trim leading/trailing whitespace
    return s.strip()


def _summarize_thread(thread: dict) -> dict:
    """Distill a Gmail thread into a compact list-row dict for the inbox UI."""
    msgs = thread.get("messages") or []
    if not msgs:
        return {"id": thread.get("id"), "subject": "", "sender": "", "snippet": "",
                "date": "", "msg_count": 0, "unread": False, "labels": []}
    first = msgs[0]
    last = msgs[-1]
    headers_first = first.get("payload", {}).get("headers", [])
    headers_last = last.get("payload", {}).get("headers", [])
    labels_union = set()
    unread = False
    for m in msgs:
        for lid in (m.get("labelIds") or []):
            labels_union.add(lid)
            if lid == "UNREAD":
                unread = True
    return {
        "id": thread.get("id"),
        "subject": _header(headers_first, "Subject") or "(no subject)",
        "sender": _header(headers_last, "From"),
        "snippet": last.get("snippet", ""),
        "date": _header(headers_last, "Date"),
        "internal_date_ms": int(last.get("internalDate") or 0),
        "msg_count": len(msgs),
        "unread": unread,
        "labels": sorted(labels_union),
        "category": categorize(sorted(labels_union)),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def list_threads(query: str = "in:inbox", max_results: int = 50,
                 include_hidden: bool = False) -> list:
    """Return a list of thread summaries matching the Gmail search query.

    By default, threads from senders in the hidden_senders block list are
    filtered out. Pass include_hidden=True to override (used for the
    "manage hidden senders" admin view).
    """
    svc = _get_gmail_service()
    res = svc.users().threads().list(
        userId="me", q=query, maxResults=max(1, min(max_results, 100)),
    ).execute()
    threads = res.get("threads", [])
    hidden = set() if include_hidden else _hidden_set()
    out = []
    for t in threads:
        full = svc.users().threads().get(
            userId="me", id=t["id"], format="metadata",
            metadataHeaders=["Subject", "From", "Date"],
        ).execute()
        summary = _summarize_thread(full)
        if hidden:
            sender_addr = _normalize_sender_email(summary.get("sender", ""))
            if sender_addr in hidden:
                continue
        out.append(summary)
    return out


def get_thread(thread_id: str) -> dict:
    """Return a fully-decoded thread for the message-detail pane."""
    svc = _get_gmail_service()
    t = svc.users().threads().get(userId="me", id=thread_id, format="full").execute()
    msgs = []
    for m in (t.get("messages") or []):
        headers = m.get("payload", {}).get("headers", [])
        body = _extract_body(m.get("payload", {}))
        msgs.append({
            "id": m.get("id"),
            "from": _header(headers, "From"),
            "to": _header(headers, "To"),
            "cc": _header(headers, "Cc"),
            "subject": _header(headers, "Subject"),
            "date": _header(headers, "Date"),
            "internal_date_ms": int(m.get("internalDate") or 0),
            "snippet": m.get("snippet", ""),
            "body_plain": body["plain"],
            "body_html": body["html"],
            "body_clean": clean_body(body["plain"] or body["html"]),
            "labels": m.get("labelIds") or [],
        })
    return {"id": t.get("id"), "messages": msgs}


# ── Hidden-sender filter (persistent block list) ─────────────────────────────
# Stored in chat_log.db so the same hide list applies to every device.

import sqlite3 as _sqlite3
from pathlib import Path as _Path


def _db_path() -> str:
    base = os.environ.get("DATA_DIR", str(_Path(__file__).resolve().parent.parent))
    return str(_Path(base) / "chat_log.db")


def _init_hidden_senders_table(cx: "_sqlite3.Connection") -> None:
    cx.execute("""
        CREATE TABLE IF NOT EXISTS inbox_hidden_senders (
            sender_email TEXT PRIMARY KEY,
            hidden_at    TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    cx.commit()


def _normalize_sender_email(sender: str) -> str:
    """Extract just the email part from 'Name <email@domain.com>' format. Lowercase."""
    if not sender:
        return ""
    import re
    m = re.search(r"<([^>]+)>", sender)
    addr = (m.group(1) if m else sender).strip().lower()
    return addr


def hide_sender(sender: str) -> dict:
    """Add a sender to the block list. Idempotent."""
    addr = _normalize_sender_email(sender)
    if not addr or "@" not in addr:
        raise ValueError(f"invalid sender: {sender!r}")
    with _sqlite3.connect(_db_path()) as cx:
        _init_hidden_senders_table(cx)
        cx.execute(
            "INSERT OR IGNORE INTO inbox_hidden_senders (sender_email) VALUES (?)",
            (addr,),
        )
        cx.commit()
    return {"hidden_email": addr}


def unhide_sender(sender: str) -> dict:
    addr = _normalize_sender_email(sender)
    with _sqlite3.connect(_db_path()) as cx:
        _init_hidden_senders_table(cx)
        cx.execute("DELETE FROM inbox_hidden_senders WHERE sender_email = ?", (addr,))
        cx.commit()
    return {"unhidden_email": addr}


def list_hidden_senders() -> list:
    with _sqlite3.connect(_db_path()) as cx:
        _init_hidden_senders_table(cx)
        rows = cx.execute(
            "SELECT sender_email, hidden_at FROM inbox_hidden_senders ORDER BY hidden_at DESC"
        ).fetchall()
    return [{"sender_email": r[0], "hidden_at": r[1]} for r in rows]


def _hidden_set() -> set:
    """Return a set of normalized hidden sender emails for fast filtering."""
    return {h["sender_email"] for h in list_hidden_senders()}


def list_recent_sent(max_results: int = 5) -> list:
    """Return a few of Glen's most recent SENT messages as voice-reference for the AI drafter."""
    svc = _get_gmail_service()
    res = svc.users().messages().list(userId="me", q="in:sent", maxResults=max_results).execute()
    out = []
    for m in (res.get("messages") or []):
        full = svc.users().messages().get(userId="me", id=m["id"], format="full").execute()
        body = _extract_body(full.get("payload", {}))
        out.append({"id": m["id"], "body": clean_body(body["plain"] or body["html"])})
    return out


def _build_reply_message(thread: dict, body: str, override_to: Optional[str] = None) -> dict:
    """Build the raw RFC-2822 message dict for a reply to the most recent message in `thread`."""
    msgs = thread.get("messages") or []
    if not msgs:
        raise ValueError("thread has no messages")
    last = msgs[-1]
    headers = last.get("payload", {}).get("headers", [])
    subject = _header(headers, "Subject")
    if subject and not subject.lower().startswith("re:"):
        subject = "Re: " + subject
    elif not subject:
        subject = "(no subject)"

    # Reply-to chain: prefer Reply-To header if present, otherwise From
    to = override_to or _header(headers, "Reply-To") or _header(headers, "From")
    msg_id_header = _header(headers, "Message-Id") or _header(headers, "Message-ID")
    references = _header(headers, "References") or msg_id_header

    mime = MIMEText(body, "plain", "utf-8")
    mime["To"] = to
    mime["Subject"] = subject
    if msg_id_header:
        mime["In-Reply-To"] = msg_id_header
    if references:
        mime["References"] = references

    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii")
    return {"raw": raw, "threadId": thread.get("id")}


def send_reply(thread_id: str, body: str, override_to: Optional[str] = None) -> dict:
    """Send a plain-text reply on `thread_id`. Returns the sent message metadata."""
    svc = _get_gmail_service()
    thread = svc.users().threads().get(userId="me", id=thread_id, format="full").execute()
    payload = _build_reply_message(thread, body, override_to=override_to)
    sent = svc.users().messages().send(userId="me", body=payload).execute()
    return {"id": sent.get("id"), "threadId": sent.get("threadId"), "labels": sent.get("labelIds", [])}


def send_email(to_email: str, subject: str, body: str, from_name: Optional[str] = None) -> dict:
    """Generic Gmail send — used by /full-report so it doesn't need SMTP creds.

    Sends as plain text from drglenswartwout@gmail.com (the authorized account).
    """
    svc = _get_gmail_service()
    mime = MIMEText(body, "plain", "utf-8")
    mime["To"] = to_email
    mime["Subject"] = subject
    if from_name:
        # Optional display name; the From address is whatever the OAuth account is
        mime["From"] = f'"{from_name}"'
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii")
    sent = svc.users().messages().send(userId="me", body={"raw": raw}).execute()
    return {"id": sent.get("id"), "threadId": sent.get("threadId")}


# ── Mutations: archive / star / read state ────────────────────────────────────

def _modify_thread(thread_id: str, add: list = None, remove: list = None) -> dict:
    svc = _get_gmail_service()
    return svc.users().threads().modify(
        userId="me", id=thread_id,
        body={"addLabelIds": add or [], "removeLabelIds": remove or []},
    ).execute()


def archive_thread(thread_id: str) -> dict:
    return _modify_thread(thread_id, remove=["INBOX"])


def star_thread(thread_id: str) -> dict:
    return _modify_thread(thread_id, add=["STARRED"])


def unstar_thread(thread_id: str) -> dict:
    return _modify_thread(thread_id, remove=["STARRED"])


def mark_read(thread_id: str) -> dict:
    return _modify_thread(thread_id, remove=["UNREAD"])


def mark_unread(thread_id: str) -> dict:
    return _modify_thread(thread_id, add=["UNREAD"])
