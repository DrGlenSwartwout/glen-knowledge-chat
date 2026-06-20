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
    """Create+approve a reveal directly and grant the free unlock, returning the
    plaintext token. Also makes the email a member so the reveal page serves the
    full payload with top_unlocked=True (Cistus Shield visible)."""
    import secrets as _s
    from dashboard import biofield_reveals
    import begin_funnel, hashlib
    token = "tkn_" + _s.token_urlsafe(8)
    th = app_module._hash_token(token)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        biofield_reveals.init_table(cx)
        biofield_reveals.init_free_unlocks(cx)
        rid, _ = biofield_reveals.upsert(
            cx, email, "2026-06-19",
            {"greeting": "Aloha", "body": "reading"},
            [{"name": "Cistus Shield", "slug": "cistus-shield", "meaning": "Calm the terrain."}], "s")
        biofield_reveals.set_token(cx, rid, th)
        biofield_reveals.approve_first(cx, rid, "glen")
        # Grant the free unlock so top_unlocked=True and the name is visible
        biofield_reveals.record_free_unlock(cx, email, rid)
        from datetime import datetime, timezone, timedelta
        cx.execute("INSERT OR IGNORE INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                   (th, email, "biofield_reveal", datetime.now(timezone.utc).isoformat(),
                    (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()))
        # Make this email a member so the reveal gate is set on GET
        begin_funnel.init_journey_tables(cx)
        sid = "entry:" + hashlib.sha1(email.strip().lower().encode()).hexdigest()[:16]
        begin_funnel.record_unlock(cx, session_id=sid, trigger="tos", email=email, tos=True)
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


# ---------------------------------------------------------------------------
# Task 3: ToS gate + one-time free-unblock ledger + reveal-top endpoint
# ---------------------------------------------------------------------------

import json as _json
import secrets as _sec
from datetime import datetime, timezone, timedelta


def _make_reveal(app_module, db, email, approve=False):
    """Create a reveal row + auth_token, return plaintext token.
    If approve=True, also sets first_approved=1."""
    from dashboard import biofield_reveals as _br
    token = "t_" + _sec.token_urlsafe(12)
    th = app_module._hash_token(token)
    with sqlite3.connect(db) as cx:
        _br.init_table(cx)
        _br.init_free_unlocks(cx)
        cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
        rid, _ = _br.upsert(cx, email, "2026-06-19",
                            {"greeting": "Aloha", "body": "Your terrain reading."},
                            [{"name": "Cistus Shield", "slug": "cistus-shield", "meaning": "Calm the terrain."},
                             {"name": "Binder Pro", "slug": "binder-pro", "meaning": "Bind and clear."}],
                            "test")
        _br.set_token(cx, rid, th)
        if approve:
            _br.approve_first(cx, rid, "glen")
        now = datetime.now(timezone.utc)
        cx.execute("INSERT OR IGNORE INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) VALUES (?,?,?,?,?)",
                   (th, email, "biofield_reveal", now.isoformat(),
                    (now + timedelta(days=30)).isoformat()))
        cx.commit()
    return token


def _make_member(db, email):
    """Set ToS so is_member(email) returns True."""
    import begin_funnel
    import hashlib
    sid = "entry:" + hashlib.sha1(email.strip().lower().encode()).hexdigest()[:16]
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
        begin_funnel.record_unlock(cx, session_id=sid, trigger="tos", email=email, tos=True)


# Test 1: GET non-member -> needs_tos, no interpretation, biofield gate NOT set
def test_get_nonmember_shows_tos_gate(monkeypatch, tmp_path):
    app_module = _load_app()
    db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    email = "nm@x.com"
    token = _make_reveal(app_module, db, email, approve=True)
    client = app_module.app.test_client()
    r = client.get(f"/begin/biofield/{token}")
    assert r.status_code == 200
    assert "no-store" in r.headers.get("Cache-Control", "")
    # __REVEAL__ payload must contain needs_tos but NOT interpretation body
    assert b"needs_tos" in r.data
    assert b"Your terrain reading" not in r.data
    # Biofield gate must NOT be set (non-member)
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email=email)
    assert "biofield" not in (st.get("unlocked_gates") or [])


# Test 2: GET member + pending (not approved) -> interpretation present, free_available false, top null, gate SET
def test_get_member_pending_shows_interpretation(monkeypatch, tmp_path):
    app_module = _load_app()
    db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    email = "mem@x.com"
    token = _make_reveal(app_module, db, email, approve=False)
    _make_member(db, email)
    client = app_module.app.test_client()
    r = client.get(f"/begin/biofield/{token}")
    assert r.status_code == 200
    # Must contain interpretation body
    assert b"Your terrain reading" in r.data
    # Extract __REVEAL__ JSON from the HTML
    data_str = r.data.decode()
    start = data_str.find("window.__REVEAL__ = ") + len("window.__REVEAL__ = ")
    end = data_str.find(";</script>", start)
    payload = _json.loads(data_str[start:end])
    assert payload["interpretation"]["greeting"] == "Aloha"
    assert payload["free_available"] is False
    assert payload["top"] is None
    assert payload["first_approved"] is False
    # Gate IS set for member
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email=email)
    assert "biofield" in (st.get("unlocked_gates") or [])


# Test 3: GET member + first_approved + free not used -> free_available true, top null
def test_get_member_approved_free_not_used(monkeypatch, tmp_path):
    app_module = _load_app()
    db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    email = "apr@x.com"
    token = _make_reveal(app_module, db, email, approve=True)
    _make_member(db, email)
    client = app_module.app.test_client()
    r = client.get(f"/begin/biofield/{token}")
    assert r.status_code == 200
    data_str = r.data.decode()
    start = data_str.find("window.__REVEAL__ = ") + len("window.__REVEAL__ = ")
    end = data_str.find(";</script>", start)
    payload = _json.loads(data_str[start:end])
    assert payload["first_approved"] is True
    assert payload["free_available"] is True
    assert payload["top"] is None  # button not yet clicked


# Test 4: POST reveal-top (member, approved, unused) -> {ok:true, top:{name...}}
# + row in biofield_free_unlocks; second POST -> {ok:false, reason:"used"}
def test_post_reveal_top_grants_once(monkeypatch, tmp_path):
    app_module = _load_app()
    db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    email = "once@x.com"
    token = _make_reveal(app_module, db, email, approve=True)
    _make_member(db, email)
    client = app_module.app.test_client()
    r1 = client.post(f"/begin/biofield/{token}/reveal-top")
    assert r1.status_code == 200
    body1 = r1.get_json()
    assert body1["ok"] is True
    assert body1["top"]["name"] == "Cistus Shield"
    assert body1["top"]["buy_url"] == "/begin/buy/cistus-shield"
    # Row must exist in ledger
    from dashboard import biofield_reveals as _br
    with sqlite3.connect(db) as cx:
        _br.init_free_unlocks(cx)
        fu_rid = _br.free_unlock_reveal_id(cx, email)
    assert fu_rid is not None
    # Second POST -> used
    r2 = client.post(f"/begin/biofield/{token}/reveal-top")
    assert r2.get_json() == {"ok": False, "reason": "used"}


# Test 5a: POST reveal-top when not approved -> reason:"pending"
def test_post_reveal_top_pending(monkeypatch, tmp_path):
    app_module = _load_app()
    db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    email = "pend@x.com"
    token = _make_reveal(app_module, db, email, approve=False)
    _make_member(db, email)
    client = app_module.app.test_client()
    r = client.post(f"/begin/biofield/{token}/reveal-top")
    assert r.get_json() == {"ok": False, "reason": "pending"}


# Test 5b: GET with no member -> POST returns reason:"tos"
def test_post_reveal_top_nonmember(monkeypatch, tmp_path):
    app_module = _load_app()
    db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    email = "nmt@x.com"
    token = _make_reveal(app_module, db, email, approve=True)
    # No _make_member call -> not a member
    client = app_module.app.test_client()
    r = client.post(f"/begin/biofield/{token}/reveal-top")
    assert r.get_json() == {"ok": False, "reason": "tos"}


# Test 6: GET member after free unlock granted on THIS reveal -> top_unlocked true, top name present
def test_get_member_after_free_unlock(monkeypatch, tmp_path):
    app_module = _load_app()
    db = _fresh(app_module, monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_send_inquiry_email", lambda *a, **k: True)
    email = "after@x.com"
    token = _make_reveal(app_module, db, email, approve=True)
    _make_member(db, email)
    client = app_module.app.test_client()
    # First grant the free unlock
    r_post = client.post(f"/begin/biofield/{token}/reveal-top")
    assert r_post.get_json()["ok"] is True
    # Now GET should show top_unlocked=true and top name present
    r_get = client.get(f"/begin/biofield/{token}")
    assert r_get.status_code == 200
    data_str = r_get.data.decode()
    start = data_str.find("window.__REVEAL__ = ") + len("window.__REVEAL__ = ")
    end = data_str.find(";</script>", start)
    payload = _json.loads(data_str[start:end])
    assert payload["top_unlocked"] is True
    assert payload["top"]["name"] == "Cistus Shield"
    # Name should be in the HTML (JSON-encoded in the script tag)
    assert b"Cistus Shield" in r_get.data


# Test 7: Invalid token GET -> friendly page, __REVEAL__ = null, no interpretation, gate not set
def test_get_invalid_token(monkeypatch, tmp_path):
    app_module = _load_app()
    db = _fresh(app_module, monkeypatch, tmp_path)
    client = app_module.app.test_client()
    r = client.get("/begin/biofield/totally-invalid-token-xyz")
    assert r.status_code == 200
    assert b"window.__REVEAL__ = null" in r.data
    assert b"Your terrain reading" not in r.data
    assert "no-store" in r.headers.get("Cache-Control", "")
    # Gate not set
    import begin_funnel
    with sqlite3.connect(db) as cx:
        st = begin_funnel.get_state(cx, email="any@x.com")
    assert "biofield" not in (st.get("unlocked_gates") or [])
