"""Gmail reply watcher — polls Glen's inbox for unread replies from
beta-cohort users, runs each through the personalization loop, and
labels processed messages so the cron is idempotent.

Public entrypoint: process_inbox_replies()
"""

import base64
import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from dashboard import gmail_token as _gmail_token


PROCESSED_LABEL = "AMG_PROCESSED"
NONUSER_LABEL = "AMG_NONUSER"

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
]


def _build_service_from_creds(creds):
    from googleapiclient.discovery import build
    return build("gmail", "v1", credentials=creds)


def _get_gmail_service(db_path=None):
    """Load durable creds (DB-first, file fallback, self-heal) and build the
    Gmail service. Returns (svc, LoadedGmail) so the caller can persist a
    refresh and record token health at the end of a run."""
    loaded = _gmail_token.load_gmail_credentials(
        db_path or _gmail_token.default_db_path(),
        name="inbox_gmail", scopes=GMAIL_SCOPES,
    )
    return _build_service_from_creds(loaded.creds), loaded


def _ensure_label(svc, label_name: str) -> str:
    """Get-or-create the named Gmail label; return its ID."""
    labels = svc.users().labels().list(userId="me").execute().get("labels", [])
    for l in labels:
        if l["name"] == label_name:
            return l["id"]
    new = svc.users().labels().create(
        userId="me",
        body={
            "name": label_name,
            "labelListVisibility": "labelHide",
            "messageListVisibility": "show",
        },
    ).execute()
    return new["id"]


def _strip_quoted_reply(body: str) -> str:
    """Remove the quoted prior message from a reply body so we feed only
    the user's new content into the LLM. Handles two common patterns:

      1. "On <date>, <name> wrote:" — strip from there onward
      2. Lines beginning with ">" — drop them
    """
    if not body:
        return ""
    # Pattern 1: "On ... wrote:" — strip from there to the end. The
    # match is non-greedy and capped to keep us from chewing through
    # legitimately long bodies.
    m = re.search(r"\n\s*On\s+.{0,200}?wrote:\s*\n?", body)
    if m:
        body = body[:m.start()]
    # Pattern 2: drop ">" prefixed lines
    out = []
    for line in body.split("\n"):
        if line.lstrip().startswith(">"):
            continue
        out.append(line)
    return "\n".join(out).strip()


def _resolve_user_id(email: str, db_path: str) -> Optional[int]:
    """Look up the users.id for the given email (case-insensitive).
    Returns None if the email isn't a registered user."""
    if not email:
        return None
    with sqlite3.connect(db_path) as cx:
        cx.row_factory = sqlite3.Row
        row = cx.execute(
            "SELECT id FROM users WHERE LOWER(email) = ?",
            (email.lower(),),
        ).fetchone()
    return row["id"] if row else None


def _record_feedback(
    db_path: str,
    user_id: Optional[int],
    raw_text: str,
    result: dict,
) -> int:
    """Persist one personal_email_feedback row and return its rowid."""
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as cx:
        cur = cx.execute(
            """INSERT INTO personal_email_feedback
                 (received_at, user_id, original_send_id, raw_text,
                  ai_summary, ai_category, routed_to,
                  extracted_topics, extracted_products, extracted_conditions)
               VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now,
                user_id,
                raw_text[:8000],
                result.get("ai_summary", ""),
                result.get("ai_category", "question"),
                result.get("routed_to", "glen-review"),
                json.dumps(result.get("extracted_topics", [])),
                json.dumps(result.get("extracted_products", [])),
                json.dumps(result.get("extracted_conditions", [])),
            ),
        )
        cx.commit()
        return cur.lastrowid


def _extract_plain_body(payload: dict) -> str:
    """Walk the MIME tree and return the first text/plain body found.
    Gmail uses urlsafe base64 with padding sometimes stripped, so we
    pad defensively before decoding."""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            try:
                return base64.urlsafe_b64decode(data + "===").decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                return ""
    for part in payload.get("parts", []) or []:
        result = _extract_plain_body(part)
        if result:
            return result
    return ""


def _extract_sender_email(headers_lower: dict) -> str:
    """Pull the bare email out of a 'From: Name <email@x.com>' header."""
    sender_raw = headers_lower.get("from", "") or ""
    m = re.search(r"<([^>]+)>", sender_raw)
    bare = (m.group(1) if m else sender_raw).strip().lower()
    return bare


def process_inbox_replies(
    svc=None,
    db_path: Optional[str] = None,
    dry_run: bool = False,
    max_messages: int = 50,
) -> dict:
    """Poll Gmail for unread inbox replies, run each through the
    personalization loop, and label processed messages.

    Idempotency: messages already carrying AMG_PROCESSED or AMG_NONUSER
    are excluded by the search query, so re-running this is safe.

    When no svc is injected, credentials are loaded from the durable store
    (dashboard/gmail_token.py) and, at the end of the run, any refreshed
    token is written back and the token health row is marked ok. An
    injected svc bypasses the token load entirely (existing callers/tests).

    Returns a counts dict:
      {"processed": int, "skipped_nonuser": int, "errored": int,
       "details": [...]}
    """
    if db_path is None:
        db_path = _gmail_token.default_db_path()

    loaded = None
    if svc is None:
        svc, loaded = _get_gmail_service(db_path)

    processed_label_id = _ensure_label(svc, PROCESSED_LABEL)
    nonuser_label_id = _ensure_label(svc, NONUSER_LABEL)
    counts = _scan_and_process(
        svc, db_path, dry_run, max_messages,
        processed_label_id, nonuser_label_id,
    )

    if loaded is not None:
        try:
            _gmail_token.persist_refreshed_credentials(db_path, loaded)
            _gmail_token.record_ok(db_path, "inbox_gmail")
        except Exception as e:  # best-effort; never fail the run on write-back
            print(f"[reply-watcher] token write-back failed: {e!r}", flush=True)

    return counts


def _scan_and_process(
    svc, db_path, dry_run, max_messages, processed_label_id, nonuser_label_id
) -> dict:
    """Query Gmail for unread, unprocessed inbox replies and run each
    through the personalization loop. Extracted from process_inbox_replies
    so the token-load/persist bookends can wrap it without touching this
    scanning logic."""
    query = (
        f"in:inbox is:unread "
        f"-label:{PROCESSED_LABEL} -label:{NONUSER_LABEL} "
        f"newer_than:7d"
    )
    resp = svc.users().messages().list(
        userId="me", q=query, maxResults=max_messages
    ).execute()
    msg_ids = [m["id"] for m in (resp.get("messages") or [])]

    counts = {
        "processed": 0,
        "skipped_nonuser": 0,
        "errored": 0,
        "details": [],
    }

    # Lazy import — keeps the module testable without anthropic configured.
    from incentive_engine import (
        process_reply,
        update_personalization_from_reply,
    )

    for mid in msg_ids:
        try:
            msg = svc.users().messages().get(
                userId="me", id=mid, format="full"
            ).execute()

            headers_lower = {
                h["name"].lower(): h["value"]
                for h in (msg.get("payload", {}).get("headers") or [])
            }
            sender = _extract_sender_email(headers_lower)

            user_id = _resolve_user_id(sender, db_path)
            if user_id is None:
                if not dry_run:
                    svc.users().messages().modify(
                        userId="me",
                        id=mid,
                        body={"addLabelIds": [nonuser_label_id]},
                    ).execute()
                counts["skipped_nonuser"] += 1
                counts["details"].append(
                    {
                        "msg_id": mid,
                        "sender": sender,
                        "action": "skipped_nonuser",
                    }
                )
                continue

            body = _extract_plain_body(msg.get("payload", {})) or msg.get(
                "snippet", ""
            )
            cleaned = _strip_quoted_reply(body)
            if not cleaned.strip():
                cleaned = msg.get("snippet", "")

            result = process_reply(
                user_id=user_id,
                original_send_id=None,  # Phase 0: no in-reply-to mapping
                raw_text=cleaned,
            )
            if not dry_run:
                _record_feedback(db_path, user_id, cleaned, result)
                update_personalization_from_reply(
                    user_id=user_id,
                    extracted_topics=result.get("extracted_topics", []),
                    extracted_products=result.get("extracted_products", []),
                )
                svc.users().messages().modify(
                    userId="me",
                    id=mid,
                    body={"addLabelIds": [processed_label_id]},
                ).execute()
            counts["processed"] += 1
            counts["details"].append(
                {
                    "msg_id": mid,
                    "sender": sender,
                    "user_id": user_id,
                    "category": result.get("ai_category"),
                    "topics": result.get("extracted_topics", []),
                }
            )
        except Exception as e:
            counts["errored"] += 1
            counts["details"].append({"msg_id": mid, "error": str(e)})

    return counts
