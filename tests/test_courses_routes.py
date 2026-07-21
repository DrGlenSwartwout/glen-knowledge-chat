import importlib
import os
import sqlite3
import pytest
from tests.courses_fixture import write_sample_course

_MHOST = "http://mentorshipu.test"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("COURSES_ROOT", str(tmp_path / "courses"))
    monkeypatch.setenv("MENTORSHIP_BASE_URL", _MHOST)
    write_sample_course(str(tmp_path / "courses"))
    import app as appmod
    importlib.reload(appmod)
    monkeypatch.setattr(appmod, "send_mentorship_setup_link", lambda *a, **k: ("test", None))
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def test_public_lesson_open_to_anon(client):
    c, _ = client
    r = c.get("/learn/ash-intro/01-intro/01-out-takes", base_url=_MHOST)
    assert r.status_code == 200
    assert b"rumble.com/embed/v1abcd" in r.data


def test_member_lesson_blocked_for_anon(client):
    c, _ = client
    r = c.get("/learn/ash-intro/01-intro/02-welcome", base_url=_MHOST)
    assert r.status_code == 403


def test_member_lesson_open_with_token(client):
    c, appmod = client
    from dashboard import course_tokens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        course_tokens.init_course_tokens_table(cx)
        token = course_tokens.mint_course_token(cx, "m@example.com", "M")
    r = c.get(f"/learn/ash-intro/01-intro/02-welcome?token={token}", base_url=_MHOST)
    assert r.status_code == 200
    assert b"rumble.com/embed/v2efgh" in r.data


def test_intake_start_ok(client):
    c, _ = client
    r = c.post("/api/mentorship/intake/start", base_url=_MHOST,
               json={"email": "new@example.com", "name": "New", "tos_agreed": True})
    assert r.status_code == 200
    assert r.get_json() == {"ok": True}


def test_intake_honeypot_silently_ok(client):
    c, _ = client
    r = c.post("/api/mentorship/intake/start", base_url=_MHOST,
               json={"email": "bot@example.com", "tos_agreed": True, "company": "x"})
    assert r.status_code == 200
    assert r.get_json() == {"ok": True}


def test_wrong_host_is_404(client):
    c, _ = client
    r = c.get("/learn/ash-intro/01-intro/01-out-takes", headers={"Host": "illtowell.com"})
    assert r.status_code == 404


def test_intake_wrong_host_is_404(client):
    c, _ = client
    r = c.post("/api/mentorship/intake/start", headers={"Host": "illtowell.com"},
               json={"email": "x@example.com", "tos_agreed": True})
    assert r.status_code == 404
