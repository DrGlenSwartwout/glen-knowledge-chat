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


def _client(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_begin_state_includes_next_step_block(monkeypatch, tmp_path):
    app_module = _load_app()
    c = _client(app_module, monkeypatch, tmp_path)
    c.set_cookie("amg_session", "ns-route-test-1")
    r = c.get("/begin/state")
    assert r.status_code == 200
    body = r.get_json()
    assert "next_step" in body
    assert "prompt" in body["next_step"] and "chips" in body["next_step"]
    # cold visitor -> opening fork
    vals = {ch.get("value") for ch in body["next_step"]["chips"] if ch["action"] == "style"}
    assert vals == {"mission", "adventure"}


def test_travel_style_route_sets_and_returns_block(monkeypatch, tmp_path):
    app_module = _load_app()
    c = _client(app_module, monkeypatch, tmp_path)
    c.set_cookie("amg_session", "ns-route-test-2")
    r = c.post("/begin/travel-style", json={"style": "adventure"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    labels = [ch["label"] for ch in body["next_step"]["chips"] if ch["role"] == "primary"]
    assert "Explore my biofield" in labels


def test_travel_style_route_rejects_bad_value(monkeypatch, tmp_path):
    app_module = _load_app()
    c = _client(app_module, monkeypatch, tmp_path)
    c.set_cookie("amg_session", "ns-route-test-3")
    r = c.post("/begin/travel-style", json={"style": "wander"})
    assert r.status_code == 400


def test_travel_style_route_threads_has_e4l_to_scan_href(monkeypatch, tmp_path):
    app_module = _load_app()
    c = _client(app_module, monkeypatch, tmp_path)
    # Force the has_e4l signal (a returning client known by email) WITHOUT
    # unlocking this session's scan gate, so voice_scan is still the first
    # undone step and its href is chosen by the signal. This asserts the route
    # actually threads signals into next_step_chips (regression guard).
    monkeypatch.setattr(app_module, "_has_e4l", lambda cx, email, state: True)
    c.set_cookie("amg_session", "ns-route-e4l")
    r = c.post("/begin/travel-style", json={"style": "adventure"})
    assert r.status_code == 200
    scan = next(ch for ch in r.get_json()["next_step"]["chips"]
                if ch["label"] == "Explore my biofield")
    assert scan["href"].startswith("https://portal.e4l.com")
