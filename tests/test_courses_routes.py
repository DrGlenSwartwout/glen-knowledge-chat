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


def test_learn_reaches_topic_pages_on_illtowell(client, monkeypatch):
    # Regression for the /learn route collision: with topic pages enabled and a
    # NON-mentorship host, /learn and /learn/<slug> must reach app.py's topic
    # handlers (learn_index / learn_topic_page), NOT the course renderer and NOT
    # a blueprint fail-closed 404. TOPIC_PAGES_ENABLED is read into a module
    # constant at import, so we flip the already-loaded module attribute (setenv
    # alone would not take effect without a reload).
    c, appmod = client
    monkeypatch.setattr(appmod, "TOPIC_PAGES_ENABLED", True)

    # /learn -> topic-page index ("Wellness Topics"), never the course catalog.
    r = c.get("/learn", base_url="http://illtowell.com")
    assert r.status_code == 200
    assert b"Wellness Topics" in r.data
    assert b"MentorshipU" not in r.data

    # /learn/<slug> -> topic handler. With no seeded page this renders the
    # "being prepared" pending page (status 200), not course content, not 404.
    r = c.get("/learn/some-unseeded-topic", base_url="http://illtowell.com")
    assert r.status_code == 200
    assert b"This guide is being prepared." in r.data
    assert b"MentorshipU" not in r.data
