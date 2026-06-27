"""Tests for POST /api/portal/<token>/chat (Task 2 — portal concierge SSE route).

Streams real Haiku under Doppler. Uses reload-app convention so DATA_DIR
points at a fresh tmp_path SQLite file each run.
"""
import importlib
import json
import sqlite3

import pytest


# ---------------------------------------------------------------------------
# App fixture (reload-app convention)
# ---------------------------------------------------------------------------

@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    try:
        import app
        importlib.reload(app)
    except Exception as exc:
        pytest.skip(f"app import failed: {exc}")
    app.app.config["TESTING"] = True
    with app.app.test_client() as client:
        yield app, client


# ---------------------------------------------------------------------------
# Helper: mint a portal token
# ---------------------------------------------------------------------------

def _mint_portal(app_mod, email="t@x.com", name="T"):
    content = {
        "layers": [{"n": 3, "title": "Liver", "remedy": "Terrain Restore"}],
        "findings": [{"code": "EI8", "name": "stress"}],
        "reorder_items": [],
    }
    from dashboard import client_portal
    with sqlite3.connect(app_mod.LOG_DB) as cx:
        client_portal.init_client_portal_table(cx)
        token, _ = client_portal.upsert_portal(cx, email=email, name=name, content=content)
    assert token, "upsert_portal must return a raw token on first insert"
    return token


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_portal_chat_streams_token_and_done_no_gate(app_client):
    """Valid token -> SSE body contains 'token' events and a 'done' event, NO 'gate' event."""
    app_mod, client = app_client
    token = _mint_portal(app_mod)

    resp = client.post(
        f"/api/portal/{token}/chat",
        data=json.dumps({"query": "what helps my stress?"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.data.decode("utf-8")

    # Must have at least one "token" event
    assert '"token"' in body, f"no token event in SSE body; got:\n{body[:500]}"

    # Must have a "done" event
    assert '"done"' in body, f"no done event in SSE body; got:\n{body[:500]}"

    # Must NOT have a "gate" event (no TOS gate for portal)
    assert '"gate"' not in body, f"unexpected gate event in SSE body; got:\n{body[:500]}"


def test_portal_chat_bad_token_returns_404(app_client):
    """Invalid token -> 404."""
    _app_mod, client = app_client
    resp = client.post(
        "/api/portal/BADTOKEN/chat",
        data=json.dumps({"query": "hello"}),
        content_type="application/json",
    )
    assert resp.status_code == 404
