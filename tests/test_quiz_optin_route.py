import importlib, sys, sqlite3, threading, time
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


def _setup_db(app_module, tmp_path, monkeypatch):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import begin_funnel, quiz_engine
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
        quiz_engine.init_quiz_tables(cx)
        cx.execute("""
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token_hash    TEXT PRIMARY KEY,
                email         TEXT NOT NULL,
                purpose       TEXT NOT NULL,
                extra         TEXT,
                created_at    TEXT NOT NULL,
                expires_at    TEXT NOT NULL,
                consumed_at   TEXT
            )
        """)
    return db


def test_optin_reaches_free_tier_tags_and_mints_guide(monkeypatch, tmp_path):
    app_module = _load_app()
    db = _setup_db(app_module, tmp_path, monkeypatch)
    captured = {}
    done = threading.Event()

    def fake_onboard(email, first="", last="", **kw):
        captured["email"] = email
        captured["tags"] = set(kw.get("extra_tags") or [])
        captured["source_tag"] = kw.get("source_tag")
        done.set()
        return {"contact_id": "x"}

    monkeypatch.setattr(app_module, "ghl_onboard_contact", fake_onboard)
    monkeypatch.setattr(app_module, "_capture_concierge_referral", lambda *a, **k: None)

    c = app_module.app.test_client()
    c.set_cookie("amg_session", "s1")
    # answer first so segment is known
    c.post("/begin/quiz/answer",
           json={"quiz_id": "eye-brain", "answers": {"q1": "watch_wait"}})
    r = c.post("/begin/quiz/opt-in",
               json={"quiz_id": "eye-brain", "name": "Ada",
                     "email": "ada@example.com", "tos": True})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["current_rung"] == "free_tier"
    assert body["redirect"] == "/begin/quiz/result"
    assert body["guide_token"]
    # email now attached to the quiz response
    import quiz_engine
    with sqlite3.connect(db) as cx:
        got = quiz_engine.get_response(cx, session_id="s1", quiz_id="eye-brain")
    assert got["email"] == "ada@example.com"
    # GHL onboard fired with the lead-magnet + segment tags
    assert done.wait(2.0)
    assert captured["email"] == "ada@example.com"
    assert "lead-magnet" in captured["tags"]
    assert "quiz-completed" in captured["tags"]
    assert "awareness:watch_wait" in captured["tags"]
    # guide token validates
    assert app_module._validate_lead_magnet_guide_link(body["guide_token"]) == "ada@example.com"


def test_optin_requires_email_and_tos(monkeypatch, tmp_path):
    app_module = _load_app()
    _setup_db(app_module, tmp_path, monkeypatch)
    c = app_module.app.test_client()
    c.set_cookie("amg_session", "s2")
    r = c.post("/begin/quiz/opt-in",
               json={"quiz_id": "eye-brain", "name": "Bo", "email": "", "tos": True})
    assert r.status_code == 400
    r2 = c.post("/begin/quiz/opt-in",
                json={"quiz_id": "eye-brain", "name": "Bo", "email": "bo@x.com", "tos": False})
    assert r2.status_code == 400
