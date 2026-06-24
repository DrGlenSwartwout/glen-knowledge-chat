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


def _seed(app_module, tmp_path, monkeypatch, answers):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import quiz_engine
    with sqlite3.connect(db) as cx:
        quiz_engine.init_quiz_tables(cx)
        quiz_engine.store_response(cx, session_id="s1", quiz_id="eye-brain", answers=answers)
    return db


def test_result_page_served():
    app_module = _load_app()
    c = app_module.app.test_client()
    r = c.get("/begin/quiz/result")
    assert r.status_code == 200


def test_result_data_open_founding_includes_card(monkeypatch, tmp_path):
    app_module = _load_app()
    _seed(app_module, tmp_path, monkeypatch, {"q1": "watch_wait", "q8": "eye_formula"})
    import dashboard.founding as founding
    monkeypatch.setattr(founding, "get_launch",
                        lambda s: {"cap": 2500, "batch_label": "Founding Batch No. 1", "closes_at": ""})
    monkeypatch.setattr(founding, "is_open", lambda cx, s, now_iso=None: True)
    monkeypatch.setattr(founding, "remaining", lambda cx, s: 1900)
    c = app_module.app.test_client()
    c.set_cookie("amg_session", "s1")
    r = c.get("/begin/quiz/result-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["taken"] is True
    assert body["profile"]["band"] == "barrier"
    assert "not been evaluated" in body["disclaimer"]
    assert body["founding"]["remaining"] == 1900
    assert body["product_url"] == "/begin/product/neuro-magnesium"
    assert body["founding"]["personal_line"]


def test_result_data_closed_founding_is_null(monkeypatch, tmp_path):
    app_module = _load_app()
    _seed(app_module, tmp_path, monkeypatch, {"q1": "general"})
    import dashboard.founding as founding
    monkeypatch.setattr(founding, "get_launch",
                        lambda s: {"cap": 2500, "batch_label": "x", "closes_at": ""})
    monkeypatch.setattr(founding, "is_open", lambda cx, s, now_iso=None: False)
    monkeypatch.setattr(founding, "remaining", lambda cx, s: 0)
    c = app_module.app.test_client()
    c.set_cookie("amg_session", "s1")
    r = c.get("/begin/quiz/result-data")
    body = r.get_json()
    assert body["founding"] is None


def test_result_data_no_answers_taken_false(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import quiz_engine
    with sqlite3.connect(db) as cx:
        quiz_engine.init_quiz_tables(cx)
    c = app_module.app.test_client()
    c.set_cookie("amg_session", "nobody")
    r = c.get("/begin/quiz/result-data")
    assert r.status_code == 200
    assert r.get_json()["taken"] is False
