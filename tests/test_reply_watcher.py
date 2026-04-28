"""Tests for reply_watcher — Gmail reply ingestion + personalization loop."""

import json
import sqlite3
from unittest.mock import MagicMock

import pytest


# ── _strip_quoted_reply ──────────────────────────────────────────────


def test_strip_quoted_reply_removes_on_wrote_pattern():
    from reply_watcher import _strip_quoted_reply

    body = (
        "Thanks for the email!\n"
        "\n"
        "I'd love more info on Terrain Restore.\n"
        "\n"
        "On Mon, Apr 28, 2026 at 7:00 AM Glen Swartwout <glen@example.com> wrote:\n"
        "> Today's discovery: tight junctions matter.\n"
        "> Try Terrain Restore."
    )
    cleaned = _strip_quoted_reply(body)
    assert "Terrain Restore" in cleaned
    assert "tight junctions" not in cleaned
    assert "Glen Swartwout" not in cleaned


def test_strip_quoted_reply_handles_arrow_prefix():
    from reply_watcher import _strip_quoted_reply

    body = "My new content.\n> previous quote\n> more quote"
    cleaned = _strip_quoted_reply(body)
    assert cleaned == "My new content."


# ── _resolve_user_id ─────────────────────────────────────────────────


def test_resolve_user_id_returns_id_for_known_email(tmp_path):
    db = str(tmp_path / "test.db")
    with sqlite3.connect(db) as cx:
        cx.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT,
                name TEXT
            );
            INSERT INTO users (id, email, name)
              VALUES (42, 'glen@example.com', 'Glen');
            """
        )
        cx.commit()
    from reply_watcher import _resolve_user_id

    assert _resolve_user_id("glen@example.com", db) == 42
    assert _resolve_user_id("Glen@Example.com", db) == 42  # case-insensitive
    assert _resolve_user_id("unknown@example.com", db) is None


# ── _record_feedback ─────────────────────────────────────────────────


def test_record_feedback_persists_all_fields(tmp_path):
    db = str(tmp_path / "test.db")
    with sqlite3.connect(db) as cx:
        cx.executescript(
            """
            CREATE TABLE personal_email_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at TEXT NOT NULL,
                user_id INTEGER,
                original_send_id INTEGER,
                raw_text TEXT,
                ai_summary TEXT,
                ai_category TEXT,
                routed_to TEXT,
                extracted_topics TEXT,
                extracted_products TEXT,
                extracted_conditions TEXT,
                glen_reviewed_at TEXT,
                action_taken TEXT
            );
            """
        )
        cx.commit()
    from reply_watcher import _record_feedback

    fid = _record_feedback(
        db,
        42,
        "Hello world",
        {
            "ai_summary": "test summary",
            "ai_category": "topic-request",
            "routed_to": "glen-review",
            "extracted_topics": ["leaky-gut"],
            "extracted_products": ["Terrain Restore"],
            "extracted_conditions": [],
        },
    )
    assert fid > 0
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        r = cx.execute(
            "SELECT * FROM personal_email_feedback WHERE id=?", (fid,)
        ).fetchone()
    assert r["user_id"] == 42
    assert r["raw_text"] == "Hello world"
    assert r["ai_category"] == "topic-request"
    assert r["routed_to"] == "glen-review"
    assert json.loads(r["extracted_topics"]) == ["leaky-gut"]
    assert json.loads(r["extracted_products"]) == ["Terrain Restore"]


# ── process_inbox_replies — dry_run ─────────────────────────────────


def _seed_min_schema(db: str):
    with sqlite3.connect(db) as cx:
        cx.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT,
                name TEXT
            );
            CREATE TABLE personal_email_state (
                user_id INTEGER PRIMARY KEY,
                topic_engagement_history TEXT,
                product_affinity TEXT,
                last_send_at TEXT,
                last_open_at TEXT,
                last_click_at TEXT
            );
            CREATE TABLE personal_email_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at TEXT NOT NULL,
                user_id INTEGER,
                original_send_id INTEGER,
                raw_text TEXT,
                ai_summary TEXT,
                ai_category TEXT,
                routed_to TEXT,
                extracted_topics TEXT,
                extracted_products TEXT,
                extracted_conditions TEXT,
                glen_reviewed_at TEXT,
                action_taken TEXT
            );
            INSERT INTO users (id, email, name)
              VALUES (42, 'a@x.com', 'Alice');
            """
        )
        cx.commit()


def _build_mock_svc(messages_resp, msg_get_resp, labels=None):
    """Build a Gmail service mock that returns the given fixtures."""
    svc = MagicMock()
    svc.users().labels().list().execute.return_value = {
        "labels": labels
        or [
            {"id": "L1", "name": "AMG_PROCESSED"},
            {"id": "L2", "name": "AMG_NONUSER"},
        ]
    }
    svc.users().messages().list().execute.return_value = messages_resp
    svc.users().messages().get().execute.return_value = msg_get_resp
    return svc


def test_process_inbox_replies_dry_run_skips_writes(tmp_path, monkeypatch):
    """dry_run must NOT write to DB or apply Gmail labels."""
    db = str(tmp_path / "test.db")
    _seed_min_schema(db)

    svc = _build_mock_svc(
        messages_resp={"messages": [{"id": "msg1"}]},
        msg_get_resp={
            "id": "msg1",
            "payload": {
                "headers": [{"name": "From", "value": "Alice <a@x.com>"}],
                "mimeType": "text/plain",
                # base64 of "Tell me about leaky gut"
                "body": {"data": "VGVsbCBtZSBhYm91dCBsZWFreSBndXQ="},
            },
            "snippet": "fallback snippet",
        },
    )

    # Stub the LLM so process_reply is hermetic
    monkeypatch.setattr(
        "incentive_engine._llm_complete",
        lambda p, max_tokens=500: json.dumps(
            {
                "summary": "User asks about leaky gut.",
                "category": "topic-request",
                "topics": ["leaky-gut"],
                "products": [],
                "conditions": [],
            }
        ),
    )

    from reply_watcher import process_inbox_replies

    counts = process_inbox_replies(svc=svc, db_path=db, dry_run=True)
    assert counts["processed"] == 1
    assert counts["skipped_nonuser"] == 0
    assert counts["errored"] == 0

    # No DB writes in dry_run
    with sqlite3.connect(db) as cx:
        n = cx.execute(
            "SELECT COUNT(*) FROM personal_email_feedback"
        ).fetchone()[0]
    assert n == 0

    # No label modifications in dry_run
    svc.users().messages().modify.assert_not_called()


def test_process_inbox_replies_unknown_sender_gets_nonuser_label(
    tmp_path, monkeypatch
):
    """Replies from senders not in users get AMG_NONUSER and are skipped."""
    db = str(tmp_path / "test.db")
    _seed_min_schema(db)

    svc = _build_mock_svc(
        messages_resp={"messages": [{"id": "msg2"}]},
        msg_get_resp={
            "id": "msg2",
            "payload": {
                "headers": [
                    {"name": "From", "value": "Stranger <stranger@x.com>"}
                ],
                "mimeType": "text/plain",
                "body": {"data": "aGVsbG8="},  # "hello"
            },
            "snippet": "hello",
        },
    )

    from reply_watcher import process_inbox_replies

    counts = process_inbox_replies(svc=svc, db_path=db, dry_run=False)
    assert counts["processed"] == 0
    assert counts["skipped_nonuser"] == 1
    assert counts["errored"] == 0

    # Verify the modify call applied the NONUSER label and NOT the PROCESSED
    # label. Filter modify() calls (not labels.list/.create chain calls).
    modify_calls = [
        c
        for c in svc.users().messages().modify.call_args_list
        if c.kwargs.get("id") == "msg2"
    ]
    assert len(modify_calls) == 1
    assert modify_calls[0].kwargs["body"] == {"addLabelIds": ["L2"]}

    # No feedback row inserted for unknown sender
    with sqlite3.connect(db) as cx:
        n = cx.execute(
            "SELECT COUNT(*) FROM personal_email_feedback"
        ).fetchone()[0]
    assert n == 0
