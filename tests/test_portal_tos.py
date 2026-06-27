"""Tests for tos_agreed in /api/portal/<token> payload and POST /api/portal/<token>/agree-tos.

Uses the reload-app convention: monkeypatches LOG_DB onto a tmp DB so the
test is isolated from prod and other tests.
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


def _app(monkeypatch, tmp_db):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        app = importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")
    monkeypatch.setattr(app, "LOG_DB", str(tmp_db))
    # init all required schema on the tmp db
    app._init_workspace_schema()
    import begin_funnel
    with sqlite3.connect(str(tmp_db)) as cx:
        begin_funnel.init_journey_tables(cx)
    return app


def _seed_portal(tmp_db):
    """Seed a portal with biofield_status 'none'; return raw token."""
    from dashboard import client_portal as cp
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.init_client_portal_table(cx)
        token, _pid = cp.upsert_portal(cx, "t@x.com", "T", {"biofield_status": "none"})
    return token


def test_tos_agreed_flow(monkeypatch, tmp_db):
    app = _app(monkeypatch, tmp_db)
    token = _seed_portal(tmp_db)
    assert token, "upsert_portal must return a raw token on first create"

    client = app.app.test_client()

    # 1. GET before TOS — tos_agreed should be False, biofield_status "none" should survive
    resp = client.get(f"/api/portal/{token}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["tos_agreed"] is False, f"Expected False before TOS, got {data.get('tos_agreed')!r}"
    assert data["biofield_status"] == "none", (
        f"biofield_status 'none' should flow through unchanged; got {data.get('biofield_status')!r}")

    # 2. POST agree-tos
    resp2 = client.post(f"/api/portal/{token}/agree-tos")
    assert resp2.status_code == 200
    d2 = resp2.get_json()
    assert d2.get("ok") is True

    # 3. Verify is_member now returns True
    assert app.is_member(email="t@x.com") is True, "is_member should be True after agree-tos"

    # 4. GET again — tos_agreed now True
    resp3 = client.get(f"/api/portal/{token}")
    assert resp3.status_code == 200
    d3 = resp3.get_json()
    assert d3["tos_agreed"] is True, f"Expected True after TOS, got {d3.get('tos_agreed')!r}"


def test_agree_tos_bad_token_returns_404(monkeypatch, tmp_db):
    app = _app(monkeypatch, tmp_db)
    client = app.app.test_client()
    resp = client.post("/api/portal/BADTOKEN/agree-tos")
    assert resp.status_code == 404
