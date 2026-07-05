"""Tests for POST /api/portal/<token>/scene-pref — the member's server-saved
fireside backdrop choice. Uses the reload-app convention (LOG_DB on a tmp DB)."""
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
    app._init_workspace_schema()
    return app


def _seed_portal(tmp_db):
    from dashboard import client_portal as cp
    with sqlite3.connect(str(tmp_db)) as cx:
        cp.init_client_portal_table(cx)
        token, _pid = cp.upsert_portal(cx, "t@x.com", "T", {"biofield_status": "none"})
    return token


def _saved(tmp_db, email):
    from dashboard import member_element_state as mes
    with sqlite3.connect(str(tmp_db)) as cx:
        row = mes.get(cx, email)
    return row["scene_override"] if row else None


def test_scene_pref_saves_and_clears(monkeypatch, tmp_db):
    app = _app(monkeypatch, tmp_db)
    token = _seed_portal(tmp_db)
    client = app.app.test_client()

    r = client.post(f"/api/portal/{token}/scene-pref", json={"element": "fire"})
    assert r.status_code == 200 and r.get_json()["element"] == "fire"
    assert _saved(tmp_db, "t@x.com") == "fire"

    # Automatic clears the override
    r2 = client.post(f"/api/portal/{token}/scene-pref", json={"element": "auto"})
    assert r2.status_code == 200 and r2.get_json()["element"] is None
    assert _saved(tmp_db, "t@x.com") is None


def test_scene_pref_rejects_garbage(monkeypatch, tmp_db):
    app = _app(monkeypatch, tmp_db)
    token = _seed_portal(tmp_db)
    client = app.app.test_client()
    r = client.post(f"/api/portal/{token}/scene-pref", json={"element": "plasma"})
    assert r.status_code == 400
    assert _saved(tmp_db, "t@x.com") is None   # nothing persisted


def test_scene_pref_bad_token_404(monkeypatch, tmp_db):
    app = _app(monkeypatch, tmp_db)
    client = app.app.test_client()
    r = client.post("/api/portal/BADTOKEN/scene-pref", json={"element": "fire"})
    assert r.status_code == 404
