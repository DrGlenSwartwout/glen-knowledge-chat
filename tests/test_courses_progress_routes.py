import importlib, sqlite3
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


def _token(appmod, email="w@x.com"):
    from dashboard import course_tokens
    with sqlite3.connect(appmod.LOG_DB) as cx:
        course_tokens.init_course_tokens_table(cx)
        return course_tokens.mint_course_token(cx, email, "W")


def test_watched_marks_for_resolved_learner(client):
    c, appmod = client
    tok = _token(appmod, "w@x.com")
    r = c.post(f"/api/courses/ash-intro/01-intro/01-out-takes/watched?token={tok}", base_url=_MHOST)
    assert r.status_code == 200 and r.get_json()["ok"] is True
    from dashboard import course_progress as cp
    with sqlite3.connect(appmod.LOG_DB) as cx:
        assert "01-out-takes" in cp.watched_lessons(cx, "w@x.com", "ash-intro", "01-intro")


def test_watched_unauthorized_without_token(client):
    c, _ = client
    r = c.post("/api/courses/ash-intro/01-intro/01-out-takes/watched", base_url=_MHOST)
    assert r.status_code == 401
