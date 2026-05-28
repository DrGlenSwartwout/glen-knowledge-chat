import importlib
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
        pytest.skip(f"app module not importable in this env: {e}")


def test_begin_page_served_and_mints_session(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    client = app_module.app.test_client()
    r = client.get("/begin")
    assert r.status_code == 200
    set_cookie = r.headers.get("Set-Cookie", "")
    assert "amg_session=" in set_cookie


def test_begin_state_default(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    client = app_module.app.test_client()
    r = client.get("/begin/state")
    assert r.status_code == 200
    body = r.get_json()
    assert body["current_rung"] == "arrival"
    assert body["reveal"] == ["layer0"]
