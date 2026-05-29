import importlib
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


def test_begin_page_served_and_mints_session(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    client = app_module.app.test_client()
    r = client.get("/begin")
    assert r.status_code == 200
    set_cookie = r.headers.get("Set-Cookie", "")
    assert "amg_session=" in set_cookie


def test_begin_state_default(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    client = app_module.app.test_client()
    r = client.get("/begin/state")
    assert r.status_code == 200
    body = r.get_json()
    assert body["current_rung"] == "arrival"
    assert body["reveal"] == ["layer0"]


def test_begin_unlock_options_200():
    app_module = _load_app()
    client = app_module.app.test_client()
    r = client.open("/begin/unlock", method="OPTIONS")
    assert r.status_code == 200


def test_begin_unlock_invalid_trigger_400(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "s1")
    r = client.post("/begin/unlock", json={"trigger": "bogus"})
    assert r.status_code == 400


def test_begin_unlock_name_then_email_tos_reaches_free_tier(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    monkeypatch.setattr(app_module, "ghl_onboard_contact",
                        lambda *a, **k: {"contact_id": "x"})
    monkeypatch.setattr(app_module, "_capture_concierge_referral",
                        lambda *a, **k: None)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "s1")
    client.post("/begin/unlock", json={"trigger": "question"})
    r = client.post("/begin/unlock", json={"trigger": "name", "name": "Ada"})
    assert r.get_json()["current_rung"] == "personalize"
    assert r.get_json()["first_name"] == "Ada"
    r = client.post("/begin/unlock", json={
        "trigger": "tos", "email": "ada@example.com", "tos": True})
    body = r.get_json()
    assert body["current_rung"] == "free_tier"
    assert "layer5" in body["reveal"]


def test_begin_unlock_onboards_once_on_free_tier_transition(monkeypatch, tmp_path):
    import sqlite3, begin_funnel, threading, time
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    lock = threading.Lock()
    calls = []

    def _rec(*a, **k):
        with lock:
            calls.append((a, k))
        return {"contact_id": "x"}

    monkeypatch.setattr(app_module, "ghl_onboard_contact", _rec)
    monkeypatch.setattr(app_module, "_capture_concierge_referral", lambda *a, **k: None)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "s9")
    # Transition into free_tier -> should onboard exactly once
    client.post("/begin/unlock", json={"trigger": "tos", "email": "z@x.com", "tos": True})
    # Wait (up to ~2s) for the daemon onboarding thread to complete
    for _ in range(40):
        with lock:
            n = len(calls)
        if n >= 1:
            break
        time.sleep(0.05)
    # A later unlock while STILL at free_tier must NOT re-onboard
    client.post("/begin/unlock", json={"trigger": "scroll"})
    time.sleep(0.3)
    with lock:
        assert len(calls) == 1, f"expected exactly one onboarding call, got {len(calls)}"


def test_begin_unlock_deep_link_returns_redirect(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "sdl")
    r = client.post("/begin/unlock", json={"trigger": "deep_link", "want": "e4l", "ref": "Jane"})
    body = r.get_json()
    assert body["awareness_stage"] == "most"
    assert body["redirect"].startswith("https://truly.vip/E4L")
    assert "utm_source=Jane" in body["redirect"]


def test_begin_unlock_deep_link_unbuilt_target_no_redirect(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "sdl2")
    r = client.post("/begin/unlock", json={"trigger": "deep_link", "want": "ash"})
    body = r.get_json()
    assert body["awareness_stage"] == "most"
    assert "redirect" not in body or body.get("redirect") in (None, "")
    assert "layer5" in body["reveal"]


def test_begin_state_returns_default_trio(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    client = app_module.app.test_client()
    r = client.get("/begin/state")
    cards = r.get_json()["surfaced_cards"]
    assert [c["key"] for c in cards] == ["quiz", "e4l_scan", "intake"]
    assert all(c.get("href") for c in cards)


def test_begin_unlock_returns_surfaced_cards(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "scards")
    r = client.post("/begin/unlock", json={"trigger": "question"})
    cards = r.get_json()["surfaced_cards"]
    assert isinstance(cards, list) and len(cards) >= 1
    assert all(c.get("key") and c.get("href") for c in cards)


def test_begin_tone_serves(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    client = app_module.app.test_client()
    r = client.get("/begin/tone")
    assert r.status_code == 200
    assert b"<html" in r.data.lower() or b"<!doctype" in r.data.lower()


def test_begin_voice_serves_and_mints_session(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    client = app_module.app.test_client()
    r = client.get("/begin/voice")
    assert r.status_code == 200
    assert "amg_session=" in r.headers.get("Set-Cookie", "")


def test_haiku_classification_guarded_below_threshold(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    calls = []
    monkeypatch.setattr(app_module, "_classify_awareness_haiku",
                        lambda *a, **k: calls.append(1))
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "sh")
    # no questions logged yet → below the >=3 threshold → no classify
    client.post("/begin/unlock", json={"trigger": "question"})
    import time; time.sleep(0.2)
    assert len(calls) == 0


def test_begin_unlock_accepts_path(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
    client = app_module.app.test_client()
    client.set_cookie("amg_session", "sp")
    r = client.post("/begin/unlock", json={"trigger": "paid_fork", "path": "pay_forward"})
    body = r.get_json()
    assert body["path"] == "pay_forward"
    assert body["current_rung"] == "choose_path"


def test_begin_path_serves_and_mints_session(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    client = app_module.app.test_client()
    r = client.get("/begin/path")
    assert r.status_code == 200
    assert "amg_session=" in r.headers.get("Set-Cookie", "")


def _seed_affiliate(db, token="tok-approved", slug="jane", status="approved"):
    import sqlite3, begin_funnel
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
        # ensure affiliate_signups exists (created by app's _init_referral_tables at import;
        # create a minimal compatible row here for the test DB)
        cx.execute("""CREATE TABLE IF NOT EXISTS affiliate_signups
            (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, organization TEXT,
             slug TEXT, status TEXT, token TEXT, created_at TEXT, short_url TEXT, referred_by TEXT)""")
        # also create the tables that affiliate_portal_data queries
        cx.execute("""CREATE TABLE IF NOT EXISTS referral_events
            (id INTEGER PRIMARY KEY AUTOINCREMENT, received_at TEXT, lead_id INTEGER,
             email TEXT, first_name TEXT, last_name TEXT, utm_source TEXT DEFAULT '',
             utm_medium TEXT DEFAULT '', utm_campaign TEXT DEFAULT '', utm_content TEXT DEFAULT '',
             utm_term TEXT DEFAULT '', quiz_score TEXT DEFAULT '', raw_json TEXT DEFAULT '')""")
        cx.execute("""CREATE TABLE IF NOT EXISTS affiliate_conversions
            (id INTEGER PRIMARY KEY AUTOINCREMENT, received_at TEXT, email TEXT,
             affiliate_slug TEXT, conversion_type TEXT, detail TEXT,
             order_value REAL, source TEXT, raw_json TEXT)""")
        cx.execute("""CREATE TABLE IF NOT EXISTS affiliate_offers
            (id INTEGER PRIMARY KEY AUTOINCREMENT, sort_order INTEGER DEFAULT 0,
             name TEXT, description TEXT DEFAULT '', url_template TEXT,
             active INTEGER DEFAULT 1, instructions TEXT DEFAULT '')""")
        cx.execute("""CREATE TABLE IF NOT EXISTS affiliate_social_links
            (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, slug TEXT, email TEXT,
             url TEXT, platform TEXT, points INTEGER, views INTEGER, likes INTEGER,
             shares INTEGER, reviewed_at TEXT)""")
        cx.execute("INSERT INTO affiliate_signups (name,email,slug,status,token,created_at) VALUES (?,?,?,?,?,?)",
                   ("Jane","jane@x.com",slug,status,token,"2026-05-28T00:00:00+00:00"))
        cx.commit()


def test_social_links_submit_approved(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    _seed_affiliate(db)
    client = app_module.app.test_client()
    r = client.post("/affiliate/social-links",
                    json={"token": "tok-approved",
                          "urls": ["https://youtu.be/abc", "not-a-url", "https://x.com/p/1"]})
    body = r.get_json()
    assert r.status_code == 200 and body["count"] == 2   # the non-url is skipped
    import sqlite3
    with sqlite3.connect(db) as cx:
        n = cx.execute("SELECT COUNT(*) FROM affiliate_social_links WHERE slug='jane'").fetchone()[0]
    assert n == 2


def test_social_links_submit_pending_or_invalid(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    _seed_affiliate(db, token="tok-pending", slug="pend", status="pending")
    client = app_module.app.test_client()
    assert client.post("/affiliate/social-links", json={"token": "tok-pending", "urls": ["https://x.com/p"]}).status_code == 403
    assert client.post("/affiliate/social-links", json={"token": "nope", "urls": ["https://x.com/p"]}).status_code == 404


def test_portal_data_returns_social_links(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    _seed_affiliate(db)
    client = app_module.app.test_client()
    client.post("/affiliate/social-links", json={"token": "tok-approved", "urls": ["https://youtu.be/abc"]})
    d = client.get("/affiliate/portal-data?token=tok-approved").get_json()
    assert "social_links" in d
    assert any(s["url"] == "https://youtu.be/abc" for s in d["social_links"])


def test_begin_ascend_serves_and_mints_session(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    client = app_module.app.test_client()
    r = client.get("/begin/ascend")
    assert r.status_code == 200
    assert "amg_session=" in r.headers.get("Set-Cookie", "")


def test_ask_serves_full_chat(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    client = app_module.app.test_client()
    r = client.get("/ask")
    assert r.status_code == 200
    expected = (app_module.STATIC / "index.html").read_bytes()
    assert r.data == expected


def test_root_serves_funnel_and_mints_session(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    client = app_module.app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert r.data == (app_module.STATIC / "begin.html").read_bytes()
    assert "amg_session=" in r.headers.get("Set-Cookie", "")


def test_begin_alias_still_serves_funnel(monkeypatch, tmp_path):
    app_module = _load_app()
    monkeypatch.setattr(app_module, "LOG_DB", str(tmp_path / "chat_log.db"))
    client = app_module.app.test_client()
    r = client.get("/begin")
    assert r.status_code == 200
    assert r.data == (app_module.STATIC / "begin.html").read_bytes()
