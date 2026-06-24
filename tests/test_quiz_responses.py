import sqlite3
import quiz_engine


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    quiz_engine.init_quiz_tables(cx)
    return cx


def test_init_is_idempotent(tmp_path):
    cx = _cx(tmp_path)
    quiz_engine.init_quiz_tables(cx)  # second call must not raise
    cols = [r[1] for r in cx.execute("PRAGMA table_info(quiz_responses)")]
    assert {"session_id", "email", "quiz_id", "answers_json", "segment"} <= set(cols)


def test_store_then_get_roundtrip(tmp_path):
    cx = _cx(tmp_path)
    rid = quiz_engine.store_response(cx, session_id="s1", quiz_id="eye-brain",
                                     answers={"q1": "watch_wait", "q2": "frequent"})
    assert rid > 0
    got = quiz_engine.get_response(cx, session_id="s1", quiz_id="eye-brain")
    assert got["answers"] == {"q1": "watch_wait", "q2": "frequent"}
    assert got["segment"] == "watch_wait"
    assert got["email"] == ""


def test_store_upserts_same_session_quiz(tmp_path):
    cx = _cx(tmp_path)
    quiz_engine.store_response(cx, session_id="s1", quiz_id="eye-brain", answers={"q1": "general"})
    quiz_engine.store_response(cx, session_id="s1", quiz_id="eye-brain",
                               answers={"q1": "family"}, email="A@B.com")
    rows = cx.execute("SELECT COUNT(*) FROM quiz_responses WHERE session_id='s1'").fetchone()[0]
    assert rows == 1
    got = quiz_engine.get_response(cx, session_id="s1", quiz_id="eye-brain")
    assert got["segment"] == "family"
    assert got["email"] == "a@b.com"


def test_get_missing_returns_none(tmp_path):
    cx = _cx(tmp_path)
    assert quiz_engine.get_response(cx, session_id="nope", quiz_id="eye-brain") is None
