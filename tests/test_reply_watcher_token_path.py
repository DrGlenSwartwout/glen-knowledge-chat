"""Tests for reply_watcher's Gmail token resolution.

Regression test for the prod bug where the reply-watcher cron returned
HTTP 500 on Render because it only ever looked for the token at
~/.config/google/token.json (via GMAIL_TOKEN_PATH or the hardcoded
default), never at /data/google-token.json — the actual location on the
Render persistent disk that the send path (dashboard/inbox.py) already
checks.

Token-path resolution (env override -> Render persistent disk -> local
default, plus the DB-first durable store) now lives entirely in
dashboard/gmail_token.py and is exercised there in tests/test_gmail_token.py
(test_loads_from_db_when_present, test_falls_back_to_file_and_self_heals_db,
test_raises_when_nowhere). reply_watcher._get_gmail_service just delegates
to gmail_token.load_gmail_credentials, so this file only checks that the
delegation surfaces GmailTokenMissing when no token exists anywhere,
without touching real Gmail/Google.
"""

import sqlite3

import pytest

from dashboard.gmail_token import GmailTokenMissing


def test_get_gmail_service_raises_when_no_token_anywhere(tmp_path, monkeypatch):
    """No usable token in the DB or on disk should surface as
    GmailTokenMissing, not a silent failure or an unrelated exception."""
    import reply_watcher

    db_path = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db_path) as cx:
        cx.execute(
            "CREATE TABLE oauth_tokens (name TEXT PRIMARY KEY, "
            "token_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        cx.commit()

    monkeypatch.delenv("GMAIL_TOKEN_PATH", raising=False)
    monkeypatch.setattr(
        "dashboard.gmail_token._FILE_CANDIDATES",
        [str(tmp_path / "nope.json")],
    )

    with pytest.raises(GmailTokenMissing):
        reply_watcher._get_gmail_service(db_path)
