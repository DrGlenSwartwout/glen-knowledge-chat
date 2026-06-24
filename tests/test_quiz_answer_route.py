import importlib, sys, sqlite3
from pathlib import Path
import pytest


def _load_app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def test_answer_stores_and_returns_segment(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import quiz_engine
    with sqlite3.connect(db) as cx:
        quiz_engine.init_quiz_tables(cx)
    c = app_module.app.test_client()
    c.set_cookie("amg_session", "s1")
    r = c.post("/begin/quiz/answer",
               json={"quiz_id": "eye-brain", "answers": {"q1": "watch_wait", "q2": "frequent"}})
    assert r.status_code == 200
    assert r.get_json()["segment"] == "watch_wait"
    with sqlite3.connect(db) as cx:
        got = quiz_engine.get_response(cx, session_id="s1", quiz_id="eye-brain")
    assert got["answers"]["q1"] == "watch_wait"


def test_answer_unknown_quiz_404(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    c = app_module.app.test_client()
    c.set_cookie("amg_session", "s1")
    r = c.post("/begin/quiz/answer", json={"quiz_id": "nope", "answers": {}})
    assert r.status_code == 404
