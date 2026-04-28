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
