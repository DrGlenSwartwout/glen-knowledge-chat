"""Guard tests for /api/access-tokens minting.

A token minted against a local/ephemeral chat_log.db never reaches the
Render persistent disk, so it silently fails to authenticate in production
(the Shaira 'Unauthorized' incident, 2026-06-04). The endpoint must refuse
to mint unless it is running against the production DB (DATA_DIR set),
unless ALLOW_LOCAL_TOKEN_MINT=1 is explicitly opted in.

Mirrors the LOG_DB / CONSOLE_SECRET monkeypatch pattern in test_household_api.py.
"""

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


def _app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")


def _setup(monkeypatch, tmp_db):
    """Patch the app onto a fresh tmp DB with workspace schema + a known admin key."""
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    app._init_workspace_schema()
    return app


def _mint(client, body=None):
    return client.post(
        "/api/access-tokens",
        headers={"X-Console-Key": "testkey", "Content-Type": "application/json"},
        json=body or {"name": "shaira", "scope": "workspace:shaira", "display_name": "Shaira"},
    )


def _token_count(tmp_db):
    with sqlite3.connect(tmp_db) as cx:
        return cx.execute("SELECT COUNT(*) FROM access_tokens").fetchone()[0]


def test_mint_blocked_on_local_run(monkeypatch, tmp_db):
    """DATA_DIR unset (local run) -> refuse, write nothing."""
    app = _setup(monkeypatch, tmp_db)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("ALLOW_LOCAL_TOKEN_MINT", raising=False)

    resp = _mint(app.app.test_client())

    assert resp.status_code == 409
    assert "production" in resp.get_json()["error"].lower()
    assert _token_count(tmp_db) == 0  # nothing persisted


def test_mint_allowed_in_production(monkeypatch, tmp_db):
    """DATA_DIR set (Render) -> mint succeeds and persists."""
    app = _setup(monkeypatch, tmp_db)
    monkeypatch.setenv("DATA_DIR", "/data")
    monkeypatch.delenv("ALLOW_LOCAL_TOKEN_MINT", raising=False)

    resp = _mint(app.app.test_client())

    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True
    assert _token_count(tmp_db) == 1


def test_local_override_allows_mint(monkeypatch, tmp_db):
    """ALLOW_LOCAL_TOKEN_MINT=1 is the explicit escape hatch for local testing."""
    app = _setup(monkeypatch, tmp_db)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.setenv("ALLOW_LOCAL_TOKEN_MINT", "1")

    resp = _mint(app.app.test_client())

    assert resp.status_code == 200
    assert _token_count(tmp_db) == 1


def test_mint_still_requires_admin_key(monkeypatch, tmp_db):
    """Guard does not weaken auth: no admin key is still 401, even in production."""
    app = _setup(monkeypatch, tmp_db)
    monkeypatch.setenv("DATA_DIR", "/data")

    resp = app.app.test_client().post(
        "/api/access-tokens",
        headers={"Content-Type": "application/json"},
        json={"name": "shaira", "scope": "workspace:shaira"},
    )

    assert resp.status_code == 401
    assert _token_count(tmp_db) == 0
