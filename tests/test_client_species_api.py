"""Console sync + read + operator override for client species."""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from dashboard import client_species as cs

HDRS = {"X-Console-Key": "testkey"}
BATCH = [{"email": "care@example.com", "species": "Cat", "animal_name": "Sasha"},
         {"email": "person@example.com", "species": "Human", "animal_name": ""}]


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


@pytest.fixture()
def client(tmp_db, monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    return app.app.test_client()


def _row(tmp_db, email):
    with sqlite3.connect(tmp_db) as cx:
        cs.init_table(cx)
        return cs.get(cx, email)


def test_sync_requires_the_key(client):
    assert client.post("/api/console/client-species/sync", json={"batch": BATCH}).status_code == 401


def test_sync_writes_rows(client, tmp_db):
    r = client.post("/api/console/client-species/sync", headers=HDRS, json={"batch": BATCH})
    assert r.status_code == 200 and r.get_json()["count"] == 2
    assert _row(tmp_db, "care@example.com")["is_animal"] is True
    assert _row(tmp_db, "person@example.com")["is_animal"] is False


def test_sync_missing_batch_is_400(client):
    assert client.post("/api/console/client-species/sync", headers=HDRS, json={}).status_code == 400


def test_sync_skips_a_bad_row(client, tmp_db):
    bad = {"batch": ["nope", {"email": "care@example.com", "species": "Cat", "animal_name": "Sasha"}]}
    r = client.post("/api/console/client-species/sync", headers=HDRS, json=bad)
    assert r.status_code == 200 and r.get_json()["count"] == 1


def test_read_no_email_returns_corpus_counts_only(client):
    client.post("/api/console/client-species/sync", headers=HDRS, json={"batch": BATCH})
    b = client.get("/api/console/client-species", headers=HDRS).get_json()
    assert set(b) == {"ok", "total", "animals"}
    assert b["total"] == 2 and b["animals"] == 1


def test_read_with_email(client):
    client.post("/api/console/client-species/sync", headers=HDRS, json={"batch": BATCH})
    b = client.get("/api/console/client-species?email=care@example.com", headers=HDRS).get_json()
    assert b["is_animal"] is True and b["animal_name"] == "Sasha"


def test_override_upserts_and_wins(client, tmp_db):
    client.post("/api/console/client-species/sync", headers=HDRS, json={"batch": BATCH})
    r = client.post("/api/console/client-species", headers=HDRS,
                    json={"email": "care@example.com", "species": "Rabbit", "animal_name": "Thumper"})
    assert r.status_code == 200 and r.get_json()["is_animal"] is True
    assert _row(tmp_db, "care@example.com")["species"] == "Rabbit"


def test_override_requires_the_key(client):
    assert client.post("/api/console/client-species",
                       json={"email": "x@y.com", "species": "Dog", "animal_name": "Rex"}).status_code == 401
