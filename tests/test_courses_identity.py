import sqlite3
from dashboard import client_portal, courses_identity as ci


def test_no_token_is_level_zero(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    assert ci.member_level_for(cx, None) == 0
    assert ci.member_level_for(cx, "") == 0


def test_valid_portal_token_is_level_one(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    client_portal.init_client_portal_table(cx)
    token = client_portal.ensure_token(cx, "learner@example.com", "Learner")
    assert token
    assert ci.member_level_for(cx, token) == 1


def test_garbage_token_is_level_zero(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    client_portal.init_client_portal_table(cx)
    assert ci.member_level_for(cx, "not-a-real-token") == 0
