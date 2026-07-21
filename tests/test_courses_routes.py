import importlib
import os
import sqlite3
import pytest
from tests.courses_fixture import write_sample_course


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("COURSES_ROOT", str(tmp_path / "courses"))
    write_sample_course(str(tmp_path / "courses"))
    import app as appmod
    importlib.reload(appmod)
    monkeypatch.setattr(appmod, "send_mentorship_setup_link", lambda *a, **k: ("test", None))
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def test_public_lesson_open_to_anon(client):
    c, _ = client
    r = c.get("/learn/ash-intro/01-intro/01-out-takes")
    assert r.status_code == 200
    assert b"rumble.com/embed/v1abcd" in r.data


def test_member_lesson_blocked_for_anon(client):
    c, _ = client
    r = c.get("/learn/ash-intro/01-intro/02-welcome")
    assert r.status_code == 403


def test_member_lesson_open_with_token(client):
    c, appmod = client
    from dashboard import client_portal
    with sqlite3.connect(appmod.LOG_DB) as cx:
        client_portal.init_client_portal_table(cx)
        token = client_portal.ensure_token(cx, "m@example.com", "M")
    r = c.get(f"/learn/ash-intro/01-intro/02-welcome?token={token}")
    assert r.status_code == 200
    assert b"rumble.com/embed/v2efgh" in r.data


def test_intake_start_returns_scoped_token(client):
    c, _ = client
    r = c.post("/api/mentorship/intake/start",
               json={"email": "new@example.com", "name": "New", "tos_agreed": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["token"]


def test_intake_honeypot_silently_ok(client):
    c, _ = client
    r = c.post("/api/mentorship/intake/start",
               json={"email": "bot@example.com", "tos_agreed": True, "company": "x"})
    assert r.status_code == 200 and r.get_json()["token"] == ""
