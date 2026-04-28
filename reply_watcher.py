"""Gmail reply watcher — polls Glen's inbox for unread replies from
beta-cohort users, runs each through the personalization loop, and
labels processed messages so the cron is idempotent.

Public entrypoint: process_inbox_replies()
"""

import base64
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


PROCESSED_LABEL = "AMG_PROCESSED"
NONUSER_LABEL = "AMG_NONUSER"

# Token written by ~/AI-Training/02 Skills/google-auth.py
DEFAULT_TOKEN_PATH = Path.home() / ".config" / "google" / "token.json"

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
]


def _get_gmail_service():
    """Build a Gmail API service client using the OAuth token written by
    `~/AI-Training/02 Skills/google-auth.py`. Token path can be overridden
    via the GMAIL_TOKEN_PATH env var (useful for Render where it lands at
    /data/google-token.json)."""
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials

    token_path = Path(os.environ.get("GMAIL_TOKEN_PATH", str(DEFAULT_TOKEN_PATH)))
    if not token_path.exists():
        raise RuntimeError(
            f"No Gmail token at {token_path}. "
            f"Run '~/AI-Training/02 Skills/google-auth.py' first."
        )
    creds = Credentials.from_authorized_user_file(
        str(token_path), scopes=GMAIL_SCOPES
    )
    return build("gmail", "v1", credentials=creds)


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

    Returns a counts dict:
      {"processed": int, "skipped_nonuser": int, "errored": int,
       "details": [...]}
    """
    if svc is None:
        svc = _get_gmail_service()
    if db_path is None:
        db_path = str(Path(__file__).parent / "chat_log.db")

    processed_label_id = _ensure_label(svc, PROCESSED_LABEL)
    nonuser_label_id = _ensure_label(svc, NONUSER_LABEL)

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
