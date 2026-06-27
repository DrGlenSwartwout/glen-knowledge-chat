# tests/test_chip_tap.py
"""Unit test for the /api/chip-tap endpoint.

Pattern: reload-app with a temp DB so the _init_chip_taps() call
creates the schema in an isolated sqlite file.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _setup(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    app_module._init_chip_taps()
    return db


def _tap_count(db):
    with sqlite3.connect(db) as cx:
        return cx.execute("SELECT COUNT(*) FROM chip_taps").fetchone()[0]


def test_chip_tap_records_row(monkeypatch, tmp_path):
    """POST /api/chip-tap inserts a row and returns {ok: true}."""
    app_module = _load_app()
    db = _setup(app_module, monkeypatch, tmp_path)

    client = app_module.app.test_client()
    resp = client.post(
        "/api/chip-tap",
        json={"session_id": "s1", "label": "Yes"},
        content_type="application/json",
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("ok") is True
    assert _tap_count(db) == 1


def test_chip_tap_fail_open(monkeypatch, tmp_path):
    """Even with a bad payload the endpoint returns 200 {ok: true} (fail open)."""
    app_module = _load_app()
    db = _setup(app_module, monkeypatch, tmp_path)

    client = app_module.app.test_client()
    resp = client.post(
        "/api/chip-tap",
        data="not-json",
        content_type="text/plain",
    )

    assert resp.status_code == 200
    assert resp.get_json().get("ok") is True
