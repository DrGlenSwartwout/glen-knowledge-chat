"""Tests for the Phase 0 incentive engine.

These tests exercise the additive schema migration that creates the
incentive engine tables alongside the existing query_log table.
"""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the worktree root is on sys.path so `import app` works when pytest
# is invoked from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_db():
    """Provide a temp SQLite DB path for tests; cleans up after."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    # Make sure the file is empty so _init_log_db() runs every CREATE fresh.
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_incentive_schema_creates_required_tables(tmp_db, monkeypatch):
    """The schema migration should create personal_email_state,
    personal_email_sends, personal_email_feedback, and
    holdout_assignments tables."""
    # `app` performs `_init_log_db()` at import time against the real
    # chat_log.db. Import it first, then redirect LOG_DB to our temp file
    # and re-run the initializer so the migration is exercised against the
    # temp DB only.
    import app

    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    app._init_log_db()

    with sqlite3.connect(tmp_db) as cx:
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
