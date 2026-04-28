"""Tests for the Phase 0 incentive engine.

These tests exercise the additive schema migration that creates the
incentive engine tables alongside the existing query_log table.
"""

import subprocess
import sqlite3
import sys
from pathlib import Path


def test_incentive_schema_creates_required_tables(tmp_path):
    """The schema migration should create personal_email_state,
    personal_email_sends, personal_email_feedback, and
    holdout_assignments tables.

    Runs in a subprocess with a fresh DATA_DIR so the assertion
    actually exercises the migration logic, not preexisting tables
    in the dev chat_log.db.
    """
    db_dir = tmp_path
    db_path = db_dir / "chat_log.db"

    repo_root = Path(__file__).resolve().parent.parent
    code = (
        "import sys, os\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        f"os.environ['DATA_DIR'] = {str(db_dir)!r}\n"
        "import app\n"
        "app._init_log_db()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"

    with sqlite3.connect(str(db_path)) as cx:
        tables = {
            r[0]
            for r in cx.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert "personal_email_state" in tables
    assert "personal_email_sends" in tables
    assert "personal_email_feedback" in tables
    assert "holdout_assignments" in tables


def test_resolve_channel_tags_includes_personal_newsletter_beta():
    """_resolve_channel_tags should produce tags for each opt-in flag."""
    from app import _resolve_channel_tags
    tags = _resolve_channel_tags(personal=True, newsletter=True, is_beta=True)
    assert "personal-email-opt-in" in tags
    assert "newsletter-opt-in" in tags
    assert "beta-personal-email" in tags
    assert "chatbot-lead" in tags  # baseline tag

def test_resolve_channel_tags_omits_unchecked():
    """If a channel is not opted in, its tag should not appear."""
    from app import _resolve_channel_tags
    tags = _resolve_channel_tags(personal=False, newsletter=True, is_beta=False)
    assert "personal-email-opt-in" not in tags
    assert "newsletter-opt-in" in tags
    assert "beta-personal-email" not in tags


# ── Task 6: content selector ─────────────────────────────────────────

import json
from datetime import datetime, timedelta, timezone


def test_select_topic_respects_anti_stale():
    """If a topic was sent in last 30 days, selector should skip it."""
    from incentive_engine import select_topic_for_user
    user_state = {
        "topic_send_history": json.dumps([
            {"topic": "wet-AMD", "last_sent_at":
             (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()},
            {"topic": "leaky-gut", "last_sent_at":
             (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()},
        ]),
        "topic_engagement_history": json.dumps([
            {"topic": "wet-AMD", "click_count": 5},
            {"topic": "leaky-gut", "click_count": 1},
        ]),
    }
    candidate_topics = ["wet-AMD", "leaky-gut", "EMF"]
    chosen = select_topic_for_user(user_state, candidate_topics, "client")
    assert chosen != "wet-AMD"  # too recent (anti-stale)
    assert chosen in ["leaky-gut", "EMF"]


def test_select_topic_prefers_high_affinity():
    """Among non-stale candidates, the one with most clicks wins."""
    from incentive_engine import select_topic_for_user
    user_state = {
        "topic_send_history": json.dumps([]),
        "topic_engagement_history": json.dumps([
            {"topic": "leaky-gut", "click_count": 7},
            {"topic": "EMF", "click_count": 1},
        ]),
    }
    chosen = select_topic_for_user(user_state, ["leaky-gut", "EMF", "vision"], "client")
    assert chosen == "leaky-gut"


def test_select_topic_returns_none_if_all_stale():
    """If every candidate was sent recently, return None."""
    from incentive_engine import select_topic_for_user
    user_state = {
        "topic_send_history": json.dumps([
            {"topic": "X", "last_sent_at":
             (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()},
            {"topic": "Y", "last_sent_at":
             (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()},
        ]),
        "topic_engagement_history": json.dumps([]),
    }
    assert select_topic_for_user(user_state, ["X", "Y"], "client") is None


def test_select_topic_deterministic_alphabetical_tiebreak():
    """When clicks tied at zero, alphabetical wins (deterministic)."""
    from incentive_engine import select_topic_for_user
    user_state = {
        "topic_send_history": json.dumps([]),
        "topic_engagement_history": json.dumps([]),
    }
    chosen = select_topic_for_user(user_state, ["zebra", "apple", "mango"], "client")
    assert chosen == "apple"


# ── Task 7: AI-draft personal email generator ────────────────────────


def test_generate_personal_email_includes_required_sections(monkeypatch):
    """Generated email must contain: teaching nugget, product link with
    coupon code, share-PS, beta banner (when in beta), unsubscribe link."""
    from incentive_engine import generate_personal_email

    fake_topic_text = "Terrain Restore activates intestinal tight junctions..."
    fake_product = {
        "name": "Terrain Restore",
        "url":  "https://truly.vip/terrain-restore",
        "code": "TR-U42-W17",
    }

    def fake_llm(prompt, max_tokens=500):
        # Subject prompt opens with "Write a short email subject line";
        # teaching prompt opens with "You are writing a short plain-text email".
        if prompt.lower().startswith("write a short email subject line"):
            return "What if your gut runs the show?"
        return ("Today's discovery: tight junctions matter. "
                "Try Terrain Restore 30 minutes before meals.")
    monkeypatch.setattr("incentive_engine._llm_complete", fake_llm)

    email = generate_personal_email(
        user={"id": 42, "name": "Test", "email": "test@example.com"},
        topic="leaky-gut",
        topic_source_text=fake_topic_text,
        product=fake_product,
        is_beta=True,
        audience="client",
    )

    assert "tight junctions" in email["body"].lower()
    assert "TR-U42-W17" in email["body"]
    assert "https://truly.vip/terrain-restore" in email["body"]
    assert "P.S." in email["body"]
    assert "Beta" in email["body"]      # banner for beta cohort
    assert "unsubscribe" in email["body"].lower()
    assert email["subject"]              # non-empty subject
