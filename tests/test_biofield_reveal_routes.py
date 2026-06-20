# tests/test_biofield_reveal_routes.py
"""Begin #4a - reveal ingest + reveal route."""
import importlib, sqlite3, sys
from pathlib import Path
import pytest


def _load_app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _fresh(app_module, monkeypatch, tmp_path):
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    from dashboard import biofield_reveals
    import begin_funnel
    with sqlite3.connect(db) as cx:
        biofield_reveals.init_table(cx)
        begin_funnel.init_journey_tables(cx)
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        cx.commit()
    monkeypatch.setattr(app_module, "ghl_onboard_contact", lambda *a, **k: {"contact_id": "x"})
    monkeypatch.setattr(app_module, "_capture_concierge_referral", lambda *a, **k: None)
    return db


def test_ingest_stores_draft_and_mints_token_once(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("CRON_SECRET", "k")
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    client = app_module.app.test_client()
    payload = {"email": "a@x.com", "scan_date": "2026-06-19",
               "interpretation": {"greeting": "Aloha", "body": "reading"},
               "remedies": [{"name": "Cistus", "slug": "cistus", "meaning": "calm"}], "source": "m"}
    r = client.post("/api/e4l/reveal-draft", json=payload, headers={"X-Cron-Secret": "k"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    r2 = client.post("/api/e4l/reveal-draft", json=payload, headers={"X-Cron-Secret": "k"})
    assert r2.status_code == 200
    from dashboard import biofield_reveals
    with sqlite3.connect(db) as cx:
        assert len(biofield_reveals.list_pending(cx)) == 1
        n = cx.execute("SELECT COUNT(*) FROM auth_tokens WHERE email='a@x.com' AND purpose='biofield_reveal'").fetchone()[0]
    assert n == 1  # token minted once


def test_ingest_requires_auth(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("CRON_SECRET", "k")
    client = app_module.app.test_client()
    r = client.post("/api/e4l/reveal-draft", json={"email": "a@x.com", "scan_date": "d",
                    "top_match": {"name": "X"}}, headers={"X-Cron-Secret": "wrong"})
    assert r.status_code == 401


def test_ingest_missing_email_400(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setenv("CRON_SECRET", "k")
    client = app_module.app.test_client()
    r = client.post("/api/e4l/reveal-draft", json={"scan_date": "d", "interpretation": {"greeting": "Hi"}},
                    headers={"X-Cron-Secret": "k"})
    assert r.status_code == 400


def test_console_list_drafts(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "ck", raising=False)
    from dashboard import biofield_reveals
    with sqlite3.connect(db) as cx:
        biofield_reveals.upsert(cx, "a@x.com", "2026-06-19",
                                {"greeting": "Aloha", "body": "reading"},
                                [{"name": "Cistus", "slug": "cistus", "meaning": "calm"}], "s")
    client = app_module.app.test_client()
    r = client.get("/api/console/biofield-reveals?key=ck")
    assert r.status_code == 200
    body = r.get_json()
    assert body["drafts"][0]["interpretation"]["greeting"] == "Aloha"
    assert body["drafts"][0]["remedies"][0]["name"] == "Cistus"


def test_console_page_served(monkeypatch, tmp_path):
    app_module = _load_app(); _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "ck", raising=False)
    client = app_module.app.test_client()
    r = client.get("/console/biofield-reveals?key=ck")
    assert r.status_code == 200
    assert b"biofield" in r.data.lower()


def _approve_a_reveal(app_module, db, email="a@x.com"):
    """Create+approve a reveal directly, returning the plaintext token."""
    import secrets as _s
    from dashboard import biofield_reveals
    token = "tkn_" + _s.token_urlsafe(8)
    th = app_module._hash_token(token)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        rid = biofield_reveals.upsert_draft(cx, email, "2026-06-19",
              {"name": "Cistus Shield", "slug": "cistus-shield", "meaning": "Calm the terrain."},
              [{"kind": "binder"}, {"kind": "mineral"}], "s")
        biofield_reveals.approve(cx, rid, "glen", th)
        from datetime import datetime, timezone, timedelta
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                   (th, email, "biofield_reveal", datetime.now(timezone.utc).isoformat(),
                    (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()))
        cx.commit()
    return token


def test_reveal_valid_token_renders_and_sets_gate(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    token = _approve_a_reveal(app_module, db)
    client = app_module.app.test_client()
    r = client.get(f"/begin/biofield/{token}")
    assert r.status_code == 200
    assert b"Cistus Shield" in r.data
    assert "no-store" in r.headers.get("Cache-Control", "")
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="a@x.com")
    assert "biofield" in st["unlocked_gates"]


def test_reveal_invalid_token_friendly_no_gate(monkeypatch, tmp_path):
    app_module = _load_app(); db = _fresh(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    r = client.get("/begin/biofield/bogus")
    assert r.status_code == 200  # friendly page, not a 500
    assert b"Cistus" not in r.data
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="a@x.com")
    assert "biofield" not in (st["unlocked_gates"] or [])
