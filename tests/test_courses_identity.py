import sqlite3
from dashboard import course_tokens, courses_identity as ci


def test_no_token_is_level_zero(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    assert ci.member_level_for(cx, None) == 0
    assert ci.member_level_for(cx, "") == 0


def test_valid_course_token_is_level_one(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    course_tokens.init_course_tokens_table(cx)
    raw = course_tokens.mint_course_token(cx, "learner@example.com", "L")
    assert ci.member_level_for(cx, raw) == 1


def test_garbage_token_is_level_zero(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    course_tokens.init_course_tokens_table(cx)
    assert ci.member_level_for(cx, "not-a-real-token") == 0
