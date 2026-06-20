# tests/test_begin_hero_identity.py
"""Begin #1 hero -- locks the identity/membership contract the hero front end
relies on: name capture by session, email+tos activation -> Tier-1 member,
session+email union to ONE record. All writes go through /begin/unlock."""

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable in this env: {e}")


def _fresh_db(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    # Free-tier transition onboards to GHL + concierge referral; neutralize both.
    monkeypatch.setattr(app_module, "ghl_onboard_contact",
                        lambda *a, **k: {"contact_id": "x"})
    monkeypatch.setattr(app_module, "_capture_concierge_referral",
                        lambda *a, **k: None)
    return db


def test_activation_makes_member_by_session_and_email(monkeypatch, tmp_path):
    app_module = _load_app()
    _fresh_db(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "hero1")

    # Name captured conversationally (trigger="name").
    client.post("/begin/unlock", json={"trigger": "name", "name": "Ada"})
    assert app_module.is_member(session_id="hero1") is False

    # Activation: email + ToS via the existing "tos" trigger.
    r = client.post("/begin/unlock", json={
        "trigger": "tos", "email": "ada@example.com", "tos": True})
    assert r.status_code == 200
    assert r.get_json()["current_rung"] == "free_tier"

    # Member now true by BOTH session and email.
    assert app_module.is_member(session_id="hero1") is True
    assert app_module.is_member(email="ada@example.com") is True

    # /begin/state reports the ToS stamp and the captured first name.
    st = client.get("/begin/state").get_json()
    assert st["tos_agreed_at"]
    assert st["first_name"] == "Ada"
    assert st["email"] == "ada@example.com"


def test_name_then_email_resolve_to_one_record(monkeypatch, tmp_path):
    app_module = _load_app()
    db = _fresh_db(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "hero2")
    client.post("/begin/unlock", json={"trigger": "name", "name": "Lee"})
    client.post("/begin/unlock", json={
        "trigger": "tos", "email": "lee@example.com", "tos": True})

    # Exactly one journey_state row carries this session; the union exposes
    # name + email + tos together.
    with sqlite3.connect(db) as cx:
        n = cx.execute(
            "SELECT COUNT(*) FROM journey_state WHERE session_id=?",
            ("hero2",)).fetchone()[0]
    assert n == 1
    import begin_funnel
    with sqlite3.connect(db) as cx:
        state = begin_funnel.get_state(cx, session_id="hero2",
                                       email="lee@example.com")
    assert state["first_name"] == "Lee"
    assert state["email"] == "lee@example.com"
    assert state["tos_agreed_at"]


def test_begin_serves_hero(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    html = app_module.app.test_client().get("/begin").get_data(as_text=True)
    assert 'class="hero"' in html
    assert 'id="hero-chat"' in html
    assert 'id="hero-messages"' in html
    assert "health goals" in html
    # The hero video is present (relocated, single instance kept in the hero).
    assert 'class="video"' in html


def test_hero_chat_scripted_greeting_and_no_feedback_controls(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    html = app_module.app.test_client().get("/begin").get_data(as_text=True)
    assert "what should I call you" in html
    assert "id=\"hero-send\"" in html
    # Hero chat surface must not render Rate / feedback controls.
    hero_start = html.index('id="hero-chat"')
    hero_end = html.index('</section>', hero_start)
    hero_block = html[hero_start:hero_end]
    assert "Rate" not in hero_block
    assert "feedback" not in hero_block.lower()


def test_hero_has_activation_and_bottom_explore(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    html = app_module.app.test_client().get("/begin").get_data(as_text=True)
    # Bottom explore block + gated link + non-member nudge.
    assert 'id="explore-bottom"' in html
    assert 'id="explore-link"' in html
    assert 'id="explore-nudge"' in html
    # Activation wiring present (the email field/markup is injected by JS, so
    # assert the JS that mints it ships in the page).
    assert "hero-activate-btn" in html
    assert "unlock('tos'" in html
    # The old top explore <p> was removed (only the bottom entry remains).
    assert html.count('href="/begin/explore"') == 1
