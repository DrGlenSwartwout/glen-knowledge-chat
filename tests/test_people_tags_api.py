"""Endpoint tests for POST /api/people/<id>/tags.

Isolated: monkeypatches app.LOG_DB to a temp sqlite db (never touches real
data) and app.CONSOLE_SECRET to a known value. Env-gated like the other api
tests (importing app builds OpenAI/Pinecone clients), so run via:
  doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat" \
    ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_people_tags_api.py
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

if not os.environ.get("PINECONE_API_KEY"):
    pytest.skip("requires app env (use doppler run)", allow_module_level=True)

import app  # noqa: E402


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    p = str(tmp_path / "chat_log.db")
    with sqlite3.connect(p) as cx:
        cx.execute(
            "CREATE TABLE people (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "email TEXT UNIQUE, tags TEXT DEFAULT '[]')"
        )
        cx.commit()
    monkeypatch.setattr(app, "LOG_DB", p)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    return p


@pytest.fixture
def client():
    return app.app.test_client()


def _seed(db_path, email, tags):
    with sqlite3.connect(db_path) as cx:
        cur = cx.execute(
            "INSERT INTO people (email, tags) VALUES (?, ?)",
            (email, json.dumps(tags)),
        )
        cx.commit()
        return cur.lastrowid


def _tags(db_path, pid):
    with sqlite3.connect(db_path) as cx:
        row = cx.execute("SELECT tags FROM people WHERE id=?", (pid,)).fetchone()
    return json.loads(row[0])


def test_endpoint_requires_key(client, db_path):
    pid = _seed(db_path, "a@b.com", ["type:client"])
    r = client.post(f"/api/people/{pid}/tags", json={"add": ["x"]})
    assert r.status_code == 401


def test_endpoint_happy_path_persists(client, db_path):
    pid = _seed(db_path, "a@b.com", ["type:client"])
    r = client.post(
        f"/api/people/{pid}/tags",
        json={"add": ["tier:pro-influencer"], "remove": []},
        headers={"X-Console-Key": "testkey"},
    )
    assert r.status_code == 200
    assert "tier:pro-influencer" in r.get_json()["tags"]
    assert "tier:pro-influencer" in _tags(db_path, pid)


def test_endpoint_remove_persists(client, db_path):
    pid = _seed(db_path, "a@b.com", ["type:client", "OD"])
    r = client.post(
        f"/api/people/{pid}/tags",
        json={"remove": ["OD"]},
        headers={"X-Console-Key": "testkey"},
    )
    assert r.status_code == 200
    assert _tags(db_path, pid) == ["type:client"]


def test_endpoint_404_unknown(client, db_path):
    r = client.post(
        "/api/people/99999999/tags",
        json={"add": ["x"]},
        headers={"X-Console-Key": "testkey"},
    )
    assert r.status_code == 404


def test_endpoint_400_malformed(client, db_path):
    pid = _seed(db_path, "a@b.com", [])
    r = client.post(
        f"/api/people/{pid}/tags",
        json={"add": "notalist"},
        headers={"X-Console-Key": "testkey"},
    )
    assert r.status_code == 400
