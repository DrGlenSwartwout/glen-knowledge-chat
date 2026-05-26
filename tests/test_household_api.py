"""Integration tests for household API endpoints + ghl_update_tags helper.

GHL calls are mocked so tests don't hit the live API. Uses the existing
`tmp_db` fixture from conftest.py + monkeypatching LOG_DB on the app
module, matching the pattern in test_full_report.py.
"""

import importlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
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


def _seed_people_schema(db_path):
    """Create the people table in the test DB with just the columns we use."""
    with sqlite3.connect(db_path) as cx:
        cx.execute("""
            CREATE TABLE people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                first_name TEXT DEFAULT '',
                last_name TEXT DEFAULT '',
                phone TEXT DEFAULT '',
                city TEXT DEFAULT '',
                state TEXT DEFAULT '',
                tags TEXT DEFAULT '[]'
            )
        """)
        cx.commit()


def _seed_household_tables(db_path):
    """Create the household tables in the test DB."""
    with sqlite3.connect(db_path) as cx:
        cx.execute("""
            CREATE TABLE households (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                head_person_id INTEGER,
                address TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_by TEXT NOT NULL
            )
        """)
        cx.execute("""
            CREATE TABLE household_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at TEXT NOT NULL,
                signal TEXT NOT NULL,
                person_ids TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                resolved_at TEXT DEFAULT '',
                resolved_by TEXT DEFAULT '',
                household_id INTEGER
            )
        """)
        cx.commit()


def _seed_person(db_path, email, first="", last="", phone="", city="", state="", tags=None):
    """Insert a person row and return its id."""
    with sqlite3.connect(db_path) as cx:
        cur = cx.execute(
            "INSERT INTO people (email, first_name, last_name, phone, city, state, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (email, first, last, phone, city, state, json.dumps(tags or []))
        )
        cx.commit()
        return cur.lastrowid


def test_ghl_update_tags_add_calls_lookup_and_put(monkeypatch, tmp_db):
    """ghl_update_tags(email, add={...}) looks up contact, merges tag set, PUTs."""
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)

    captured = {}

    def fake_ghl_get(path, params=None):
        captured["get_path"] = path
        captured["get_params"] = params
        return {"contacts": [{"id": "C123", "email": "test@x.com", "tags": ["existing"]}]}, None

    def fake_ghl_put(path, payload):
        captured["put_path"] = path
        captured["put_payload"] = payload
        return {}, None

    monkeypatch.setattr(app, "_ghl_get", fake_ghl_get)
    monkeypatch.setattr(app, "_ghl_put", fake_ghl_put)
    monkeypatch.setattr(app, "GHL_API_KEY", "fake-key")

    contact_id, err = app.ghl_update_tags("test@x.com", add={"household:smith"})
    assert err is None
    assert contact_id == "C123"
    assert captured["get_path"] == "/contacts/lookup"
    assert captured["put_path"] == "/contacts/C123"
    assert set(captured["put_payload"]["tags"]) == {"existing", "household:smith"}


def test_ghl_update_tags_remove_subtracts_from_existing(monkeypatch, tmp_db):
    """ghl_update_tags(email, remove={...}) subtracts tags before PUT."""
    app = _app()

    def fake_ghl_get(path, params=None):
        return {"contacts": [{"id": "C456", "email": "test@x.com", "tags": ["keep", "household:old"]}]}, None

    captured = {}
    def fake_ghl_put(path, payload):
        captured["payload"] = payload
        return {}, None

    monkeypatch.setattr(app, "_ghl_get", fake_ghl_get)
    monkeypatch.setattr(app, "_ghl_put", fake_ghl_put)
    monkeypatch.setattr(app, "GHL_API_KEY", "fake-key")

    contact_id, err = app.ghl_update_tags("test@x.com", remove={"household:old"})
    assert err is None
    assert contact_id == "C456"
    assert set(captured["payload"]["tags"]) == {"keep"}


def test_ghl_update_tags_falls_through_to_upsert_when_no_contact(monkeypatch, tmp_db):
    """If lookup returns empty, fall through to ghl_upsert_contact so the
    contact gets created with the add tags."""
    app = _app()

    def fake_ghl_get(path, params=None):
        return {"contacts": []}, None

    captured = {}
    def fake_upsert(email, first_name="", last_name="", phone="", source_tag="", extra_tags=None):
        captured["upsert_call"] = {"email": email, "extra_tags": list(extra_tags or [])}
        return "C789", True, None

    monkeypatch.setattr(app, "_ghl_get", fake_ghl_get)
    monkeypatch.setattr(app, "ghl_upsert_contact", fake_upsert)
    monkeypatch.setattr(app, "GHL_API_KEY", "fake-key")

    contact_id, err = app.ghl_update_tags("new@x.com", add={"household:smith"})
    assert err is None
    assert contact_id == "C789"
    assert captured["upsert_call"]["email"] == "new@x.com"
    assert "household:smith" in captured["upsert_call"]["extra_tags"]
