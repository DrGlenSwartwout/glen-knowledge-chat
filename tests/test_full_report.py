"""Tests for the Phase 2B /full-report endpoint.

These tests exercise the routing surface (400 + 404 paths). The happy
path (LLM regeneration + email send) requires Anthropic + Pinecone +
SMTP and is verified by live integration after deploy, not in CI.
"""

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


def _load_app():
    """Import the app module (skip the test if not importable)."""
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable in this env: {e}")


def test_full_report_endpoint_requires_log_id():
    """POST /full-report without log_id must return 400."""
    app_module = _load_app()
    client = app_module.app.test_client()
    r = client.post("/full-report", json={})
    assert r.status_code == 400
    body = r.get_json() or {}
    assert "log_id" in (body.get("error") or "").lower()


def test_full_report_endpoint_404_for_unknown_log_id(monkeypatch, tmp_path):
    """log_id not present in query_log must return 404 (no LLM call)."""
    app_module = _load_app()

    db = tmp_path / "test.db"
    with sqlite3.connect(str(db)) as cx:
        cx.execute(
            """CREATE TABLE query_log (
                id INTEGER PRIMARY KEY,
                query TEXT,
                level TEXT,
                session_id TEXT,
                email_sent_at TEXT
            )"""
        )
        cx.commit()

    # Point the route at the empty test DB
    monkeypatch.setattr(app_module, "LOG_DB", str(db))

    client = app_module.app.test_client()
    r = client.post("/full-report", json={"log_id": 9999})
    assert r.status_code == 404
    body = r.get_json() or {}
    assert "not found" in (body.get("error") or "").lower()


def test_full_report_options_returns_200():
    """CORS preflight (OPTIONS) should short-circuit with 200 before
    any DB/LLM access."""
    app_module = _load_app()
    client = app_module.app.test_client()
    r = client.open("/full-report", method="OPTIONS")
    assert r.status_code == 200


def test_email_sent_at_column_in_query_log_schema(tmp_path):
    """The Phase 2B migration must add an email_sent_at column to
    query_log so /full-report can stamp it on email send.
    """
    import subprocess

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
        cols = {r[1] for r in cx.execute("PRAGMA table_info(query_log)")}
    assert "email_sent_at" in cols
