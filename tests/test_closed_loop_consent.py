"""Closed-loop consent: a GHL DND/unsubscribe (carried via console_push) revokes
consent:opted-in in the People hub through _upsert_person_additive."""
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest


def _app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture
def app_db(monkeypatch, tmp_path):
    app = _app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    app._init_people_table()
    return app, db


def _seed(db, email, tags):
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO people (email, tags, created_at, updated_at) VALUES (?,?,?,?)",
                   (email, json.dumps(tags), "", ""))
        cx.commit()


def _tags(db, email):
    with sqlite3.connect(db) as cx:
        return set(json.loads(cx.execute("SELECT tags FROM people WHERE email=?", (email,)).fetchone()[0]))


def test_dnd_flag_revokes_optin(app_db):
    app, db = app_db
    _seed(db, "u@x.com", ["type:client", "consent:opted-in"])
    with sqlite3.connect(db) as cx:
        app._upsert_person_additive(cx, {"email": "u@x.com", "dnd": True})
        cx.commit()
    t = _tags(db, "u@x.com")
    assert "consent:opted-in" not in t and "consent:unsubscribed" in t
    assert "type:client" in t  # other tags preserved


def test_email_bounced_tag_revokes(app_db):
    app, db = app_db
    _seed(db, "b@x.com", ["type:client", "consent:opted-in"])
    with sqlite3.connect(db) as cx:
        app._upsert_person_additive(cx, {"email": "b@x.com", "tags": ["email bounced"]})
        cx.commit()
    t = _tags(db, "b@x.com")
    assert "consent:opted-in" not in t and "consent:unsubscribed" in t


def test_sms_unsubscribe_does_not_revoke_email(app_db):
    app, db = app_db
    _seed(db, "s@x.com", ["type:client", "consent:opted-in"])
    with sqlite3.connect(db) as cx:
        app._upsert_person_additive(cx, {"email": "s@x.com", "tags": ["unsubscribed on sms"]})
        cx.commit()
    t = _tags(db, "s@x.com")
    assert "consent:opted-in" in t and "consent:unsubscribed" not in t


def test_normal_sync_does_not_revoke(app_db):
    app, db = app_db
    _seed(db, "ok@x.com", ["type:client", "consent:opted-in"])
    with sqlite3.connect(db) as cx:
        app._upsert_person_additive(cx, {"email": "ok@x.com", "dnd": False, "tags": ["nes client"]})
        cx.commit()
    t = _tags(db, "ok@x.com")
    assert "consent:opted-in" in t and "consent:unsubscribed" not in t


def test_new_contact_with_dnd_inserts_unsubscribed(app_db):
    app, db = app_db
    with sqlite3.connect(db) as cx:
        app._upsert_person_additive(cx, {"email": "new@x.com", "tags": ["consent:opted-in"], "dnd": True})
        cx.commit()
    t = _tags(db, "new@x.com")
    assert "consent:unsubscribed" in t and "consent:opted-in" not in t


def test_via_merge_endpoint(app_db):
    app, db = app_db
    _seed(db, "e@x.com", ["type:client", "consent:opted-in"])
    c = app.app.test_client()
    r = c.post("/api/people?merge_tags=1", json=[{"email": "e@x.com", "dnd": True}],
               headers={"X-Console-Key": "testkey"})
    assert r.status_code == 200
    t = _tags(db, "e@x.com")
    assert "consent:opted-in" not in t and "consent:unsubscribed" in t
