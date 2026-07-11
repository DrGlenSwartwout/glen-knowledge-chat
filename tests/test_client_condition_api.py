"""Slice 3 console API: GET/POST /api/console/client-condition.
Console-key gated (mirrors test_support_programs_api.py)."""
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

HDRS = {"X-Console-Key": "testkey"}


def _app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")


@pytest.fixture()
def app_mod(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    app._init_people_table()
    return app


def _seed_person(db, email, conditions=None, tags=None):
    with sqlite3.connect(db) as cx:
        cx.execute(
            "INSERT INTO people (email, conditions, tags, created_at, updated_at) "
            "VALUES (?,?,?,?,?)",
            (email, json.dumps(conditions or []), json.dumps(tags or []), "", ""))
        cx.commit()


# ---------------------------------------------------------------------------
# GET /api/console/client-condition
# ---------------------------------------------------------------------------

def test_get_client_condition_requires_console_key(app_mod):
    client = app_mod.app.test_client()
    r = client.get("/api/console/client-condition?email=jane@example.com")
    assert r.status_code in (401, 403)


def test_get_client_condition_requires_email(app_mod):
    client = app_mod.app.test_client()
    r = client.get("/api/console/client-condition", headers=HDRS)
    assert r.status_code == 400


def test_get_client_condition_auto_detects_from_conditions(app_mod, tmp_db):
    _seed_person(tmp_db, "jane@example.com", conditions=["Wet AMD"])
    client = app_mod.app.test_client()
    r = client.get("/api/console/client-condition?email=jane@example.com", headers=HDRS)
    assert r.status_code == 200
    j = r.get_json()
    assert j["email"] == "jane@example.com"
    assert j["auto_detected"] == "wet-amd"
    assert j["resolved"] == "wet-amd"
    assert j["override"] is None
    assert "Wet AMD" in j["tags"]


def test_get_client_condition_no_match_returns_nulls(app_mod, tmp_db):
    _seed_person(tmp_db, "jane@example.com")
    client = app_mod.app.test_client()
    r = client.get("/api/console/client-condition?email=jane@example.com", headers=HDRS)
    assert r.status_code == 200
    j = r.get_json()
    assert j["resolved"] is None
    assert j["override"] is None
    assert j["auto_detected"] is None


def test_get_client_condition_unknown_email_returns_nulls(app_mod):
    client = app_mod.app.test_client()
    r = client.get("/api/console/client-condition?email=nobody@x.com", headers=HDRS)
    assert r.status_code == 200
    j = r.get_json()
    assert j["resolved"] is None
    assert j["override"] is None
    assert j["auto_detected"] is None
    assert j["tags"] == []


# ---------------------------------------------------------------------------
# POST /api/console/client-condition
# ---------------------------------------------------------------------------

def test_post_client_condition_requires_console_key(app_mod):
    client = app_mod.app.test_client()
    r = client.post("/api/console/client-condition",
                     json={"email": "jane@example.com", "condition_key": "wet-amd"})
    assert r.status_code in (401, 403)


def test_post_client_condition_requires_email(app_mod):
    client = app_mod.app.test_client()
    r = client.post("/api/console/client-condition", headers=HDRS,
                     json={"condition_key": "wet-amd"})
    assert r.status_code == 400


def test_post_client_condition_rejects_bogus_key(app_mod):
    client = app_mod.app.test_client()
    r = client.post("/api/console/client-condition", headers=HDRS,
                     json={"email": "jane@example.com", "condition_key": "not-a-real-key"})
    assert r.status_code == 400


def test_post_client_condition_sets_override_visible_via_get(app_mod, tmp_db):
    _seed_person(tmp_db, "jane@example.com", conditions=["Wet AMD"])
    client = app_mod.app.test_client()
    r = client.post("/api/console/client-condition", headers=HDRS,
                     json={"email": "jane@example.com", "condition_key": "dry-eye"})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True
    assert j["resolved"] == "dry-eye"

    r2 = client.get("/api/console/client-condition?email=jane@example.com", headers=HDRS)
    j2 = r2.get_json()
    assert j2["override"] == "dry-eye"
    assert j2["auto_detected"] == "wet-amd"
    assert j2["resolved"] == "dry-eye"  # override wins over auto-detected


def test_post_client_condition_clears_override_with_empty_key(app_mod, tmp_db):
    _seed_person(tmp_db, "jane@example.com", conditions=["Wet AMD"])
    client = app_mod.app.test_client()
    client.post("/api/console/client-condition", headers=HDRS,
                json={"email": "jane@example.com", "condition_key": "dry-eye"})
    r = client.post("/api/console/client-condition", headers=HDRS,
                     json={"email": "jane@example.com", "condition_key": ""})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True
    assert j["resolved"] == "wet-amd"  # falls back to auto-detected

    r2 = client.get("/api/console/client-condition?email=jane@example.com", headers=HDRS)
    assert r2.get_json()["override"] is None


def test_post_client_condition_clears_override_with_null_key(app_mod, tmp_db):
    _seed_person(tmp_db, "jane@example.com")
    client = app_mod.app.test_client()
    client.post("/api/console/client-condition", headers=HDRS,
                json={"email": "jane@example.com", "condition_key": "wet-amd"})
    r = client.post("/api/console/client-condition", headers=HDRS,
                     json={"email": "jane@example.com", "condition_key": None})
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True
    assert j["resolved"] is None
