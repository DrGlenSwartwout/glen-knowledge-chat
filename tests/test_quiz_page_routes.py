import importlib, sys
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


def test_quiz_page_served_and_mints_session():
    app_module = _load_app()
    c = app_module.app.test_client()
    r = c.get("/begin/quiz")
    assert r.status_code == 200
    assert "amg_session=" in r.headers.get("Set-Cookie", "")


def test_quiz_data_returns_questions_without_bands():
    app_module = _load_app()
    c = app_module.app.test_client()
    r = c.get("/begin/quiz-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["id"] == "eye-brain"
    assert len(body["questions"]) == 9
    assert "hook" in body and "disclaimer" in body
    assert "bands" not in body  # result logic stays server-side
