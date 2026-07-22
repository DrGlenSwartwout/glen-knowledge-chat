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
    assert b"<h2>Out-takes</h2>" in r.data


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
    assert b"youtube.com/embed/v2efgh" in r.data
    assert b"<h2>Welcome</h2>" in r.data


def test_member_lesson_body_has_script_stripped(client, tmp_path):
    # Stage 1.5: lesson bodies are stored HTML from Practice Better. A script
    # tag in the source content must never reach a rendered lesson page.
    c, appmod = client
    from dashboard import course_tokens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        course_tokens.init_course_tokens_table(cx)
        token = course_tokens.mint_course_token(cx, "s@example.com", "S")
    lesson_path = tmp_path / "courses" / "ash-intro" / "01-intro" / "02-welcome.md"
    lesson_path.write_text(
        "---\ntitle: Welcome\naccess: member\ndownloads: []\n---\n"
        '<h2>Welcome</h2><script>alert(document.cookie)</script>'
        "<p>Welcome transcript here.</p>\n"
    )
    r = c.get(f"/learn/ash-intro/01-intro/02-welcome?token={token}", base_url=_MHOST)
    assert r.status_code == 200
    # The page shell legitimately ships its own <script src="..."> asset tags;
    # what must NEVER survive is the inline script from the lesson body.
    assert b"alert(document.cookie)" not in r.data
    assert b"<script>alert" not in r.data
    assert b"Welcome transcript" in r.data


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


def test_intake_rate_limited_after_burst(client):
    c, _ = client
    ok = 0
    for i in range(12):
        r = c.post("/api/mentorship/intake/start", base_url=_MHOST,
                   json={"email": f"user{i}@example.com", "tos_agreed": True})
        if r.status_code == 200:
            ok += 1
    # Same IP burst: after the per-IP cap, further attempts are rejected (429).
    r = c.post("/api/mentorship/intake/start", base_url=_MHOST,
               json={"email": "last@example.com", "tos_agreed": True})
    assert r.status_code == 429


def test_mentorship_host_learn_serves_course_catalog(client):
    # Forward-delegation coverage: on the MENTORSHIP host, app.py's learn_index
    # delegates to the blueprint's learn_home() rather than serving topic pages.
    c, _ = client
    r = c.get("/learn", base_url=_MHOST)
    assert r.status_code == 200
    assert b"MentorshipU" in r.data  # course catalog h1, via app.py -> learn_home()


def test_mentorship_host_course_home_serves_course(client):
    # Forward-delegation coverage: on the MENTORSHIP host, app.py's
    # learn_topic_page delegates to the blueprint's course_home() rather than
    # serving a topic page.
    c, _ = client
    r = c.get("/learn/ash-intro", base_url=_MHOST)
    assert r.status_code == 200
    assert b"ASH Intro" in r.data  # course title, via app.py -> course_home()


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


def test_learn_home_shows_registration_form_for_anon(client):
    # H6: /learn renders an inline form (no separate /learn/register route)
    # that posts straight to the working intake API.
    c, _ = client
    r = c.get("/learn", base_url=_MHOST)
    assert r.status_code == 200
    assert b'id="register"' in r.data
    assert b"/api/mentorship/intake/start" in r.data
    assert b'type="email"' in r.data
    assert b'name="tos_agreed"' in r.data
    assert b'name="company"' in r.data  # honeypot


def test_learn_home_hides_registration_form_for_member(client):
    c, appmod = client
    from dashboard import course_tokens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        course_tokens.init_course_tokens_table(cx)
        token = course_tokens.mint_course_token(cx, "m2@example.com", "M2")
    r = c.get(f"/learn?token={token}", base_url=_MHOST)
    # learn_home redirects after consuming a ?token= link; follow it to land
    # on the clean /learn page with the member cookie set.
    r = c.get("/learn", base_url=_MHOST, headers={"Cookie": f"mu_token={token}"})
    assert r.status_code == 200
    assert b'id="register"' not in r.data


def test_learn_register_dead_link_is_gone_everywhere(client):
    # No CTA anywhere may still point at the dead /learn/register route:
    # learn_home's own CTA, the locked_register messaging on a course page,
    # and the locked_register messaging on a member-gated lesson page.
    c, _ = client

    r = c.get("/learn", base_url=_MHOST)
    assert r.status_code == 200
    assert b"/learn/register" not in r.data
    assert b"/learn#register" in r.data

    r = c.get("/learn/ash-intro", base_url=_MHOST)
    assert r.status_code == 200
    assert b"/learn/register" not in r.data
    assert b"/learn#register" in r.data

    r = c.get("/learn/ash-intro/01-intro/02-welcome", base_url=_MHOST)
    assert r.status_code == 403
    assert b"/learn/register" not in r.data
    assert b"/learn#register" in r.data
