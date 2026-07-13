"""Durable Gmail OAuth token loading for the reply-watcher and console inbox.

Source of truth is the oauth_tokens DB row (name="inbox_gmail"), matching the
pattern app.py:_run_cron already uses for glen_gmail/rae_gmail. Falls back to the
token file on the Render persistent disk and self-heals the DB from it. Standalone:
takes db_path, never imports app.py.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections import namedtuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Sequence

DEFAULT_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# Token-file fallback, same order as reply_watcher/dashboard.inbox used before.
_FILE_CANDIDATES = [
    "/data/google-token.json",                                  # Render persistent disk
    str(Path.home() / ".config" / "google" / "token.json"),     # local dev
]

_lock = threading.Lock()

LoadedGmail = namedtuple("LoadedGmail", ["creds", "source", "original_json", "name"])


class GmailTokenMissing(RuntimeError):
    """Raised when no usable Gmail token exists in the DB or on disk."""


def default_db_path() -> str:
    # dashboard/ is one level below the repo root where chat_log.db lives.
    return str(Path(__file__).resolve().parent.parent / "chat_log.db")


def _read_db_token(db_path: str, name: str) -> Optional[str]:
    with sqlite3.connect(db_path, timeout=10) as cx:
        row = cx.execute(
            "SELECT token_json FROM oauth_tokens WHERE name=?", (name,)
        ).fetchone()
    return row[0] if row else None


def _write_db_token(db_path: str, name: str, token_json: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with _lock, sqlite3.connect(db_path, timeout=10) as cx:
        cx.execute(
            "CREATE TABLE IF NOT EXISTS oauth_tokens (name TEXT PRIMARY KEY, "
            "token_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        cx.execute(
            "INSERT INTO oauth_tokens (name, token_json, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET token_json=excluded.token_json, "
            "updated_at=excluded.updated_at",
            (name, token_json, ts),
        )
        cx.commit()


def _read_file_token() -> Optional[str]:
    env = os.environ.get("GMAIL_TOKEN_PATH")
    for c in ([env] if env else []) + _FILE_CANDIDATES:
        if c and Path(c).exists():
            return Path(c).read_text()
    return None


def _build_creds(token_json: str, scopes: Sequence[str]):
    from google.oauth2.credentials import Credentials
    info = json.loads(token_json)
    granted = set(info.get("scopes") or [])
    requested = set(scopes or DEFAULT_SCOPES)
    # Pass the intersection so a refresh never asks for a scope the token lacks
    # (which Google rejects as invalid_scope). Mirrors the old inbox.py logic.
    effective = list(requested & granted) if granted else list(requested)
    if not effective:
        effective = list(granted) or list(requested)
    return Credentials.from_authorized_user_info(info, scopes=effective)


def load_gmail_credentials(db_path: str, name: str = "inbox_gmail",
                           scopes: Optional[Sequence[str]] = None) -> LoadedGmail:
    scopes = list(scopes or DEFAULT_SCOPES)
    token_json = _read_db_token(db_path, name)
    source = "db"
    if not token_json:
        token_json = _read_file_token()
        source = "file"
    if not token_json:
        raise GmailTokenMissing(
            f"No Gmail token for '{name}' in the oauth_tokens DB or on disk. "
            f"Re-run '~/AI-Training/02 Skills/google-auth.py' and "
            f"PUT it to /api/tokens/{name}."
        )
    creds = _build_creds(token_json, scopes)
    normalized = creds.to_json()  # canonical baseline for refresh comparison
    if source == "file":
        _write_db_token(db_path, name, normalized)  # self-heal the durable store
    return LoadedGmail(creds=creds, source=source, original_json=normalized, name=name)


def persist_refreshed_credentials(db_path: str, loaded: LoadedGmail) -> bool:
    """Write the token back to the DB if google-auth refreshed it during the run.
    Best-effort: comparison is against the normalized baseline captured at load."""
    current = loaded.creds.to_json()
    if current == loaded.original_json:
        return False
    _write_db_token(db_path, loaded.name, current)
    return True


def _health_name(name: str) -> str:
    """Return the health row name for a token name."""
    return f"{name}_health"


def _parse_iso(s: str) -> datetime:
    """Parse an ISO 8601 timestamp string."""
    return datetime.fromisoformat(s)


def record_ok(db_path: str, name: str, now_iso: Optional[str] = None) -> None:
    """Record that the token is healthy. Clears alert dedup window."""
    now = now_iso or datetime.now(timezone.utc).isoformat()
    _write_db_token(db_path, _health_name(name),
                    json.dumps({"healthy": True, "last_ok": now, "last_alert": None}))


def record_alert(db_path: str, name: str, now_iso: str) -> None:
    """Record that an alert occurred for this token."""
    raw = _read_db_token(db_path, _health_name(name))
    state = json.loads(raw) if raw else {}
    state["healthy"] = False
    state["last_alert"] = now_iso
    _write_db_token(db_path, _health_name(name), json.dumps(state))


def should_send_alert(db_path: str, name: str, now_iso: str,
                      window_hours: int = 6) -> bool:
    """Check if an alert should be sent based on the dedup window.

    Returns True if:
    - No prior health row exists (first alert)
    - No prior alert was recorded
    - Enough time has passed since the last alert (outside the window)
    """
    raw = _read_db_token(db_path, _health_name(name))
    if not raw:
        return True
    last_alert = (json.loads(raw) or {}).get("last_alert")
    if not last_alert:
        return True
    return _parse_iso(now_iso) - _parse_iso(last_alert) >= timedelta(hours=window_hours)
