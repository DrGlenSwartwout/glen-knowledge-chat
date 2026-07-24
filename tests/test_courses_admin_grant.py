import sqlite3
import time

import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    # Do NOT importlib.reload(app): a reload with DATA_DIR set re-runs app.py's
    # module bootstrap, which starts a BackgroundScheduler + prewarm daemon that
    # are never shut down and leak into the rest of the suite (timing-
    # nondeterministic bystander failures on CI). Redirect the module globals we
    # need directly instead — the proven pattern used by tests/test_support_program_*.
    import app as m
    monkeypatch.setattr(m, "LOG_DB", tmp_path / "chat_log.db")
    monkeypatch.setattr(m, "CONSOLE_SECRET", "sekret")  # module global read by _console_key_ok
    monkeypatch.setattr(m, "send_mentorship_setup_link", lambda *a, **k: ("test", None))
    m.app.config["TESTING"] = True
    return m


def _level(appmod, email, now):
    from dashboard import course_entitlements as ce
    with sqlite3.connect(appmod.LOG_DB) as cx:
        ce.init_course_entitlements_table(cx)
        return ce.paid_level_for(cx, email, now=now)


def test_grant_requires_console_key(appmod):
    r = appmod.app.test_client().post("/console/courses/grant-membership",
                                      json={"email": "c@x.com", "months": 1})
    assert r.status_code == 401


def test_grant_with_key_lifts_to_level_2(appmod):
    r = appmod.app.test_client().post("/console/courses/grant-membership",
                                      json={"email": "C@x.com", "months": 2},
                                      headers={"X-Console-Key": "sekret"})
    assert r.status_code == 200
    assert _level(appmod, "c@x.com", now=time.time() + 10) == 2
    assert _level(appmod, "c@x.com", now=time.time() + 70 * 86400) == 0  # past 2 months


def test_grant_bad_input_400(appmod):
    r = appmod.app.test_client().post("/console/courses/grant-membership",
                                      json={"email": "", "months": 1},
                                      headers={"X-Console-Key": "sekret"})
    assert r.status_code == 400


def test_grant_explicit_zero_months_rejected(appmod):
    # An explicit months:0 must be a 400 with NO grant, not silently promoted to 1.
    r = appmod.app.test_client().post("/console/courses/grant-membership",
                                      json={"email": "z@x.com", "months": 0},
                                      headers={"X-Console-Key": "sekret"})
    assert r.status_code == 400
    assert _level(appmod, "z@x.com", now=time.time() + 10) == 0
