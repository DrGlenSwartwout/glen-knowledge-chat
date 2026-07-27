"""Build #2a Plan 2, Task 5: per-module drip gate on the paid lesson route +
the self-select unlock-next endpoint. Uses the same fixture course as
tests/test_courses_routes.py: 'ash-intro', paid module '03-pro' with a
single lesson '01-advanced'."""
import importlib
import sqlite3
import time

import pytest
from tests.courses_fixture import write_sample_course

_MHOST = "http://mentorshipu.test"
_PAID_URL = "/learn/ash-intro/03-pro/01-advanced"
_COURSE = "ash-intro"
_MODULE = "03-pro"
_LESSON = "01-advanced"


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


def _grant_membership(appmod, email, active=True):
    from dashboard import course_entitlements as ce
    with sqlite3.connect(appmod.LOG_DB) as cx:
        ce.init_course_entitlements_table(cx)
        until = (time.time() + 3600) if active else (time.time() - 3600)
        ce.grant_membership(cx, email, until_epoch=until, source="stripe", stripe_ref=f"sub_{email}")


def _unlock(appmod, email, course=_COURSE, module=_MODULE):
    from dashboard import course_module_unlocks as cmu
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cmu.unlock_module(cx, email, course, module)


def _complete_module(appmod, email, course=_COURSE, module=_MODULE, lessons=(_LESSON,)):
    from dashboard import course_progress as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        for lesson in lessons:
            cp.mark_watched(cx, email, course, module, lesson)
        cp.record_homework(cx, email, course, module, "my reflection")


def test_full_cert_holder_opens_paid_lesson(client):
    c, appmod = client
    tok = _token(appmod, "cert@x.com")
    _grant_cert(appmod, "cert@x.com")
    r = c.get(f"{_PAID_URL}?token={tok}", base_url=_MHOST)
    assert r.status_code == 200
    assert "Advanced transcript here" in r.get_data(as_text=True)


def test_drip_member_unlocked_and_active_opens(client):
    c, appmod = client
    tok = _token(appmod, "drip@x.com")
    _grant_membership(appmod, "drip@x.com", active=True)
    _unlock(appmod, "drip@x.com")
    r = c.get(f"{_PAID_URL}?token={tok}", base_url=_MHOST)
    assert r.status_code == 200
    assert "Advanced transcript here" in r.get_data(as_text=True)


def test_drip_member_unlocked_but_membership_lapsed_403(client):
    c, appmod = client
    tok = _token(appmod, "lapsed@x.com")
    _grant_membership(appmod, "lapsed@x.com", active=False)
    _unlock(appmod, "lapsed@x.com")
    r = c.get(f"{_PAID_URL}?token={tok}", base_url=_MHOST)
    body = r.get_data(as_text=True)
    assert r.status_code == 403
    assert "Advanced transcript here" not in body
    assert "<script>alert(1)</script>" not in body


def test_drip_member_active_but_module_not_unlocked_403(client):
    c, appmod = client
    tok = _token(appmod, "notunlocked@x.com")
    _grant_membership(appmod, "notunlocked@x.com", active=True)
    r = c.get(f"{_PAID_URL}?token={tok}", base_url=_MHOST)
    body = r.get_data(as_text=True)
    assert r.status_code == 403
    assert "Advanced transcript here" not in body


def test_completed_module_banked_even_when_membership_lapsed(client):
    c, appmod = client
    tok = _token(appmod, "banked@x.com")
    _complete_module(appmod, "banked@x.com")
    _grant_membership(appmod, "banked@x.com", active=False)
    r = c.get(f"{_PAID_URL}?token={tok}", base_url=_MHOST)
    assert r.status_code == 200
    assert "Advanced transcript here" in r.get_data(as_text=True)


def test_anonymous_blocked_from_paid_lesson(client):
    c, _ = client
    r = c.get(_PAID_URL, base_url=_MHOST)
    body = r.get_data(as_text=True)
    assert r.status_code == 403
    assert "Advanced transcript here" not in body


def test_unlock_next_sets_pref_with_token(client):
    c, appmod = client
    tok = _token(appmod, "picker@x.com")
    r = c.post(f"/api/courses/{_COURSE}/unlock-next?token={tok}", json={"module": _MODULE}, base_url=_MHOST)
    assert r.status_code == 200
    assert r.get_json() == {"ok": True}
    from dashboard import course_module_unlocks as cmu
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert cmu.take_unlock_pref(cx, "picker@x.com", _COURSE) == _MODULE


def test_unlock_next_unauthorized_without_token(client):
    c, _ = client
    r = c.post(f"/api/courses/{_COURSE}/unlock-next", json={"module": _MODULE}, base_url=_MHOST)
    assert r.status_code == 401


def test_unlock_next_invalid_module_400(client):
    c, appmod = client
    tok = _token(appmod, "badmod@x.com")
    r = c.post(f"/api/courses/{_COURSE}/unlock-next?token={tok}", json={"module": "nope-does-not-exist"},
               base_url=_MHOST)
    assert r.status_code == 400


def test_enroll_panel_shows_discount_copy(client, monkeypatch):
    c, appmod = client
    tok = _token(appmod, "shopper@x.com")
    r = c.get(f"{_PAID_URL}?token={tok}", base_url=_MHOST)
    body = r.get_data(as_text=True)
    assert r.status_code == 403
    assert "save $567" in body  # $2,997 vs $297x12=$3,564 -> $567 discount, sells the one-time option


def test_course_home_shows_locked_paid_module_with_enroll_panel(client):
    c, appmod = client
    tok = _token(appmod, "browsercase@x.com")
    r = c.get(f"/learn/{_COURSE}?token={tok}", base_url=_MHOST)
    body = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "(certification module)" in body
    assert "Pay in full" in body  # enroll panel shown
    assert f"/learn/{_COURSE}/{_MODULE}/{_LESSON}" not in body  # no link when locked


def test_course_home_shows_unlocked_paid_module_as_link(client):
    c, appmod = client
    tok = _token(appmod, "browserunlocked@x.com")
    _grant_membership(appmod, "browserunlocked@x.com", active=True)
    _unlock(appmod, "browserunlocked@x.com")
    r = c.get(f"/learn/{_COURSE}?token={tok}", base_url=_MHOST)
    body = r.get_data(as_text=True)
    assert r.status_code == 200
    assert f'<a href="/learn/{_COURSE}/{_MODULE}/{_LESSON}">' in body
    assert "(certification module)" not in body
