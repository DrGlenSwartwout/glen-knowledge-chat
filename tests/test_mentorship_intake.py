import sqlite3
from dashboard import mentorship_intake as mi


def test_create_and_resolve_session(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    mi.init_mentorship_sessions_table(cx)
    token = mi.create_session(cx, "s@example.com", "Sam", now=1000.0)
    assert token
    assert mi.resolve_session(cx, token, now=1001.0) == "s@example.com"


def test_expired_session_resolves_none(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    mi.init_mentorship_sessions_table(cx)
    token = mi.create_session(cx, "s@example.com", "Sam", now=1000.0)
    later = 1000.0 + 72 * 3600 + 1
    assert mi.resolve_session(cx, token, now=later) is None


def test_unknown_token_resolves_none(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    mi.init_mentorship_sessions_table(cx)
    assert mi.resolve_session(cx, "nope", now=1.0) is None
