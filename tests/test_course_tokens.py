import sqlite3
from dashboard import course_tokens as ctok


def test_mint_then_resolve(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    ctok.init_course_tokens_table(cx)
    raw = ctok.mint_course_token(cx, "learner@example.com", "Learner")
    assert raw and isinstance(raw, str)
    assert ctok.resolve_course_token(cx, raw) == "learner@example.com"


def test_raw_token_not_stored(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    ctok.init_course_tokens_table(cx)
    raw = ctok.mint_course_token(cx, "learner@example.com", "L")
    rows = cx.execute("SELECT token_hash FROM course_tokens").fetchall()
    assert rows and rows[0][0] != raw  # only the hash is stored


def test_unknown_token_resolves_none(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    ctok.init_course_tokens_table(cx)
    assert ctok.resolve_course_token(cx, "nope") is None
    assert ctok.resolve_course_token(cx, "") is None
