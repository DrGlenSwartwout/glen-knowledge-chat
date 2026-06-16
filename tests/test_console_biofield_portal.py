# tests/test_console_biofield_portal.py
import sqlite3

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    appmod._init_auth_tables()
    monkeypatch.setattr(appmod, "CONSOLE_SECRET", "test-secret")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


_LAYER = {"n": 1, "title": "Calm", "meaning": "settle", "remedy": "Terrain Restore",
          "dosing": "10 drops 3x/day"}


def test_post_creates_portal_and_returns_url(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "x@y.com", "name": "X",
                     "content": {"greeting": "Aloha", "layers": [_LAYER]}})
    assert r.status_code == 200
    j = r.get_json()
    assert j["token"] and j["url"].endswith(j["token"])
    # content round-trips through the public portal API
    r2 = c.get(f"/api/portal/{j['token']}")
    assert r2.get_json()["layers"][0]["title"] == "Calm"


def test_post_requires_console_key(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal",
               json={"email": "x@y.com", "content": {"layers": [_LAYER]}})
    assert r.status_code == 401


def test_post_requires_email(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"content": {"layers": [_LAYER]}})
    assert r.status_code == 400


def test_post_requires_some_content(client):
    c, _ = client
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "x@y.com", "name": "X", "content": {}})
    assert r.status_code == 400


def test_post_send_emails_link(client, monkeypatch):
    c, appmod = client
    sent = {}
    monkeypatch.setattr(appmod, "_send_full_report_email",
                        lambda to, name, subj, body, **k: sent.update(to=to, body=body))
    r = c.post("/api/console/biofield-portal?key=test-secret",
               json={"email": "e@y.com", "name": "E", "send": True,
                     "content": {"greeting": "hi", "layers": [_LAYER]}})
    tok = r.get_json()["token"]
    assert sent["to"] == "e@y.com" and tok in sent["body"]
