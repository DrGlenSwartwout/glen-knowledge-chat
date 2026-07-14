"""GET /api/portal/<token> — the `life_stress` payload key (Task 5).

Mirrors tests/test_support_program_payload.py's shape:
  - flag off              -> no `life_stress` key; `life_stress_enabled` falsy
  - flag on, no reco      -> no `life_stress` key; `life_stress_enabled` present
  - flag on + reco        -> `life_stress` present, verbatim from _life_stress_for
  - best-effort             a builder error never breaks the rest of the payload
"""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import client_portal as cp

EMAIL = "lscaregiver@example.com"

RECO = {
    "label": "Life Stress",
    "patterns": [{"emotion": "Fear", "score": 1.0}],
    "items": [{"name": "Mimulus Flower Essence",
               "url": "/begin/product/mimulus-flower-essence-in-terrain-restore",
               "note": "for the fear pattern in your scan"}],
}


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture()
def app_env(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.delenv("LIFE_STRESS_ENABLED", raising=False)
    with sqlite3.connect(tmp_db) as cx:
        cx.row_factory = sqlite3.Row
        cp.init_client_portal_table(cx)
        token, _pid = cp.upsert_portal(cx, EMAIL, "Caregiver", {})
    client = app.app.test_client()
    return app, client, token


def test_flag_off_no_life_stress_key(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setattr(app, "_life_stress_for", lambda email: dict(RECO))
    j = client.get(f"/api/portal/{token}").get_json()
    assert "life_stress" not in j
    assert not j.get("life_stress_enabled")


def test_flag_on_enabled_flag_always_present(app_env, monkeypatch):
    app, client, token = app_env
    j = client.get(f"/api/portal/{token}").get_json()
    assert not j.get("life_stress_enabled")

    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    j = client.get(f"/api/portal/{token}").get_json()
    assert j["life_stress_enabled"] is True


def test_flag_on_with_recommendation_returns_block(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    monkeypatch.setattr(app, "_life_stress_for", lambda email: dict(RECO))

    j = client.get(f"/api/portal/{token}").get_json()
    assert "life_stress" in j
    assert j["life_stress"] == RECO


def test_flag_on_no_recommendation_no_key(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")
    monkeypatch.setattr(app, "_life_stress_for", lambda email: None)

    j = client.get(f"/api/portal/{token}").get_json()
    assert "life_stress" not in j
    assert j["life_stress_enabled"] is True


def test_builder_error_does_not_break_payload(app_env, monkeypatch):
    app, client, token = app_env
    monkeypatch.setenv("LIFE_STRESS_ENABLED", "1")

    def _boom(email):
        raise RuntimeError("boom")

    monkeypatch.setattr(app, "_life_stress_for", _boom)
    j = client.get(f"/api/portal/{token}").get_json()
    assert "life_stress" not in j
    assert j["life_stress_enabled"] is True
    # rest of the payload still present
    assert "email" in j or "practitioner_brand" in j
