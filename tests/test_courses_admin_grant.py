import importlib
import sqlite3
import time

import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CONSOLE_SECRET", "sekret")
    import app as m
    importlib.reload(m)
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
