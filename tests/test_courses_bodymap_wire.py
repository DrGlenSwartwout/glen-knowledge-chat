"""Plan 3 Task 4: the ash-certification course's Body module (02-body) is wired
to the interactive Body-Map homework tool (static/course-bodymap.js) instead of
the plain free-text textarea. Every other module keeps the textarea path.

Uses the REAL course content under courses/ash-certification (no COURSES_ROOT
override -- courses_content.courses_root() falls back to the repo's own
courses/ dir), so this exercises the real 02-body / 03-mind module slugs and
real lesson slugs.
"""
import importlib
import json
import sqlite3

import pytest

_MHOST = "http://mentorshipu.test"
_COURSE = "ash-certification"
_BODY_MODULE = "02-body"
_BODY_LESSON = "01-minding-body-1"
_MIND_MODULE = "03-mind"
_MIND_LESSON = "01-mind-mapping-1"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MENTORSHIP_BASE_URL", _MHOST)
    monkeypatch.delenv("COURSES_ROOT", raising=False)  # use the real courses/ dir
    import app as appmod
    importlib.reload(appmod)
    monkeypatch.setattr(appmod, "send_mentorship_setup_link", lambda *a, **k: ("test", None))
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), appmod


def _token(appmod, email):
    from dashboard import course_tokens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        course_tokens.init_course_tokens_table(cx)
        return course_tokens.mint_course_token(cx, email, "T")


def _grant_cert(appmod, email):
    from dashboard import course_entitlements as ce
    with sqlite3.connect(appmod.LOG_DB) as cx:
        ce.init_course_entitlements_table(cx)
        ce.grant_cert(cx, email, source="stripe", stripe_ref=f"cs_{email}")


def _organs_zone():
    import bodymap_store
    return bodymap_store.zone_ids("organs")[0]


def test_serves_course_bodymap_js(client):
    c, _ = client
    r = c.get("/course-bodymap.js", base_url=_MHOST)
    assert r.status_code == 200
    assert "CourseBodyMap" in r.get_data(as_text=True)


def test_body_lesson_uses_bodymap_widget(client):
    c, appmod = client
    email = "body-learner@x.com"
    tok = _token(appmod, email)
    _grant_cert(appmod, email)
    r = c.get(f"/learn/{_COURSE}/{_BODY_MODULE}/{_BODY_LESSON}?token={tok}", base_url=_MHOST)
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "course-bodymap.js" in body
    assert 'id="mu-hw-bodymap"' in body
    assert 'id="mu-hw"' not in body


def test_other_module_still_uses_textarea(client):
    c, appmod = client
    email = "mind-learner@x.com"
    tok = _token(appmod, email)
    _grant_cert(appmod, email)
    r = c.get(f"/learn/{_COURSE}/{_MIND_MODULE}/{_MIND_LESSON}?token={tok}", base_url=_MHOST)
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert 'id="mu-hw"' in body
    assert "course-bodymap.js" not in body


def test_valid_bodymap_homework_submission_succeeds(client, monkeypatch):
    c, appmod = client
    from dashboard import homework_analysis
    monkeypatch.setattr(homework_analysis, "analyze",
                        lambda module, assignment, submission: {"rating": "Good", "feedback": "Nice."})
    email = "bodyhw@x.com"
    tok = _token(appmod, email)
    _grant_cert(appmod, email)
    zone = _organs_zone()
    payload = json.dumps({"system": "organs", "marks": [{"zone": zone, "note": "n"}], "note": ""})
    r = c.post(f"/api/courses/{_COURSE}/{_BODY_MODULE}/homework?token={tok}",
               json={"payload": payload}, base_url=_MHOST)
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    from dashboard import course_progress as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        hw = cp.homework(cx, email, _COURSE, _BODY_MODULE)
    assert hw is not None and hw["payload"] == payload


def test_empty_bodymap_homework_submission_400(client):
    c, appmod = client
    email = "emptybodyhw@x.com"
    tok = _token(appmod, email)
    _grant_cert(appmod, email)
    payload = json.dumps({"system": "organs", "marks": [], "note": ""})
    r = c.post(f"/api/courses/{_COURSE}/{_BODY_MODULE}/homework?token={tok}",
               json={"payload": payload}, base_url=_MHOST)
    assert r.status_code == 400
