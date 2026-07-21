from __future__ import annotations

import hashlib
import secrets
import time


def _hash(token: str) -> str:
    return hashlib.sha256((token or "").strip().encode("utf-8")).hexdigest()


def init_course_tokens_table(cx) -> None:
    cx.execute(
        "CREATE TABLE IF NOT EXISTS course_tokens("
        "token_hash TEXT PRIMARY KEY, email TEXT, name TEXT, created_at TEXT)"
    )
    cx.commit()


def mint_course_token(cx, email: str, name: str = "") -> str:
    init_course_tokens_table(cx)
    raw = secrets.token_urlsafe(32)
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cx.execute(
        "INSERT INTO course_tokens(token_hash, email, name, created_at) VALUES(?,?,?,?)",
        (_hash(raw), (email or "").strip().lower(), name, created),
    )
    cx.commit()
    return raw


def resolve_course_token(cx, token: str) -> str | None:
    if not token:
        return None
    row = cx.execute(
        "SELECT email FROM course_tokens WHERE token_hash=?", (_hash(token),)
    ).fetchone()
    return row[0] if row else None
