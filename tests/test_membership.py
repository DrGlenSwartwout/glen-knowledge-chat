import pytest
import sqlite3
from datetime import datetime, timedelta

from tests.test_begin_routes import _load_app


def test_init_membership_tables_idempotent_creates_three_tables(tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    # call twice; must not raise; must create all three tables
    app_module.init_membership_tables(cx)
    app_module.init_membership_tables(cx)
    names = {r[0] for r in cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    expected = {"memberships", "escalation_queue", "studio_credit_intents"}
    assert expected.issubset(names)


def test_init_membership_tables_creates_expected_columns(tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    cx = sqlite3.connect(db)
    app_module.init_membership_tables(cx)
    mem_cols = {r[1] for r in cx.execute("PRAGMA table_info(memberships)").fetchall()}
    assert {"id","email","granted_at","expires_at","granted_by","source",
            "truly_vip_ref","notes","last_reminder_at"}.issubset(mem_cols)
    esc_cols = {r[1] for r in cx.execute("PRAGMA table_info(escalation_queue)").fetchall()}
    assert {"id","created_at","email","query_text","ai_response","ai_confidence",
            "flag_reason","status","glen_reply_url","glen_reply_text","replied_at"}.issubset(esc_cols)
    sci_cols = {r[1] for r in cx.execute("PRAGMA table_info(studio_credit_intents)").fetchall()}
    assert {"id","created_at","email","studio_ref","notes"}.issubset(sci_cols)


@pytest.fixture
def app_module_with_db(monkeypatch, tmp_path):
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    cx = sqlite3.connect(db)
    # auth_tokens exists from existing init; ensure it does
    try:
        import begin_funnel
        begin_funnel.init_journey_tables(cx)
    except Exception: pass
    # Ensure auth_tokens table exists in the test DB (mirror app.py:145 schema)
    cx.executescript("""
        CREATE TABLE IF NOT EXISTS auth_tokens (
          token_hash TEXT PRIMARY KEY,
          email      TEXT,
          purpose    TEXT NOT NULL,
          extra      TEXT,
          created_at TEXT NOT NULL,
          expires_at TEXT NOT NULL,
          consumed_at TEXT
        );
    """)
    app_module.init_membership_tables(cx)
    cx.close()
    return app_module, db


def test_mint_returns_plaintext_token_and_stores_hash(app_module_with_db):
    app_module, db = app_module_with_db
    plain = app_module._mint_membership_magic_link("user@example.com")
    assert isinstance(plain, str) and len(plain) > 20
    cx = sqlite3.connect(db)
    th = app_module._hash_token(plain)
    row = cx.execute(
        "SELECT email, purpose, consumed_at FROM auth_tokens WHERE token_hash=?",
        (th,)
    ).fetchone()
    assert row is not None
    assert row[0] == "user@example.com"
    assert row[1] == "membership_magic_link"
    assert row[2] is None


def test_validate_happy_path_returns_email(app_module_with_db):
    app_module, db = app_module_with_db
    plain = app_module._mint_membership_magic_link("happy@example.com")
    email = app_module._validate_membership_magic_link(plain)
    assert email == "happy@example.com"


def test_validate_returns_none_when_expired(app_module_with_db):
    app_module, db = app_module_with_db
    plain = app_module._mint_membership_magic_link("expired@example.com", ttl_min=-1)
    email = app_module._validate_membership_magic_link(plain)
    assert email is None


def test_validate_returns_none_when_consumed(app_module_with_db):
    app_module, db = app_module_with_db
    plain = app_module._mint_membership_magic_link("once@example.com")
    th = app_module._hash_token(plain)
    cx = sqlite3.connect(db)
    cx.execute("UPDATE auth_tokens SET consumed_at=? WHERE token_hash=?",
               (datetime.utcnow().isoformat()+"Z", th))
    cx.commit(); cx.close()
    email = app_module._validate_membership_magic_link(plain)
    assert email is None


def test_validate_returns_none_when_purpose_mismatch(app_module_with_db):
    app_module, db = app_module_with_db
    # Manually insert a token with a different purpose; the validator should reject.
    import secrets
    plain = secrets.token_urlsafe(32)
    th = app_module._hash_token(plain)
    now_iso = datetime.utcnow().isoformat()+"Z"
    exp_iso = (datetime.utcnow()+timedelta(minutes=15)).isoformat()+"Z"
    cx = sqlite3.connect(db)
    cx.execute(
        "INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
        "VALUES (?,?,?,?,?,?)",
        (th, "wrong@example.com", "other_purpose", "{}", now_iso, exp_iso))
    cx.commit(); cx.close()
    assert app_module._validate_membership_magic_link(plain) is None


# ── Slice 2: admin grant + admin escalations listing ─────────────────────────

import smtplib
import json as _json
from tests.test_inquiry import FakeSMTP


@pytest.fixture
def fake_smtp(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "u@e.com")
    monkeypatch.setenv("SMTP_PASS", "p")
    monkeypatch.setenv("SMTP_FROM", "hello@remedymatch.com")
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    FakeSMTP.instances = []
    yield FakeSMTP


@pytest.fixture
def app_client_mem(monkeypatch, tmp_path, fake_smtp):
    """Flask test client with mem-init done and CONSOLE_SECRET set."""
    app_module = _load_app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app_module, "LOG_DB", db)
    # Patch both app.CONSOLE_SECRET and dashboard.CONSOLE_SECRET (the decorator
    # reads the dashboard module's copy, not app.py's).
    import dashboard as _dashboard
    monkeypatch.setattr(app_module, "CONSOLE_SECRET", "test-console-secret-xyz")
    monkeypatch.setattr(_dashboard, "CONSOLE_SECRET", "test-console-secret-xyz")
    cx = sqlite3.connect(db)
    # auth_tokens (mirror app.py:145 schema)
    cx.executescript("""
        CREATE TABLE IF NOT EXISTS auth_tokens (
          token_hash TEXT PRIMARY KEY,
          email      TEXT,
          purpose    TEXT NOT NULL,
          extra      TEXT,
          created_at TEXT NOT NULL,
          expires_at TEXT NOT NULL,
          consumed_at TEXT
        );
    """)
    # journey_events
    try:
        import begin_funnel
        begin_funnel.init_journey_tables(cx)
    except Exception: pass
    app_module.init_membership_tables(cx)
    cx.close()
    return app_module.app.test_client(), app_module, db


def _ckey():
    return {"X-Console-Key": "test-console-secret-xyz"}


def test_grant_requires_console_key(app_client_mem):
    client, _, _ = app_client_mem
    r = client.post("/admin/membership/grant",
                    json={"email":"x@e.com","source":"video"})
    assert r.status_code in (401, 403)


def test_grant_happy_path_inserts_row_sends_email_logs_event(app_client_mem, fake_smtp):
    client, app_module, db = app_client_mem
    r = client.post("/admin/membership/grant",
                    json={"email":"jane@example.com","source":"video",
                          "truly_vip_ref":"https://truly.vip/Results/abc"},
                    headers=_ckey())
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert "membership_id" in body
    assert "magic_link_url" in body
    assert body["magic_link_url"].endswith(  # ends with the /coaching/auth/<token>
        body["magic_link_url"].split("/coaching/auth/")[-1])
    assert "/coaching/auth/" in body["magic_link_url"]
    assert "expires_at" in body
    # memberships row
    cx = sqlite3.connect(db)
    mrow = cx.execute(
        "SELECT email, source, truly_vip_ref, granted_by FROM memberships WHERE id=?",
        (body["membership_id"],)
    ).fetchone()
    assert mrow == ("jane@example.com", "video", "https://truly.vip/Results/abc", "glen") or \
           (mrow[0] == "jane@example.com" and mrow[1] == "video" and mrow[2] == "https://truly.vip/Results/abc")
    # FakeSMTP captured the magic-link email TO the member
    sends = [s for inst in fake_smtp.instances for s in inst.sent]
    assert any("jane@example.com" in to for (_, to, _) in sends)
    # journey_events row
    je = cx.execute(
        "SELECT detail FROM journey_events WHERE trigger='membership_granted'"
    ).fetchall()
    assert len(je) == 1
    d = _json.loads(je[0][0])
    assert d.get("source") == "video"
    assert d.get("days") == 30
    cx.close()


def test_grant_rejects_unknown_source(app_client_mem):
    client, _, _ = app_client_mem
    r = client.post("/admin/membership/grant",
                    json={"email":"x@e.com","source":"bogus_source"},
                    headers=_ckey())
    assert r.status_code == 400


def test_grant_rejects_missing_email(app_client_mem):
    client, _, _ = app_client_mem
    r = client.post("/admin/membership/grant",
                    json={"source":"video"},
                    headers=_ckey())
    assert r.status_code == 400


def test_grant_default_30_days_when_days_unset(app_client_mem):
    client, app_module, db = app_client_mem
    r = client.post("/admin/membership/grant",
                    json={"email":"d30@example.com","source":"video"},
                    headers=_ckey())
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    granted_at, expires_at = cx.execute(
        "SELECT granted_at, expires_at FROM memberships WHERE email=?",
        ("d30@example.com",)
    ).fetchone()
    g = datetime.fromisoformat(granted_at.rstrip("Z"))
    e = datetime.fromisoformat(expires_at.rstrip("Z"))
    # ~30 days within a few seconds
    delta = (e - g).total_seconds()
    assert 30*86400 - 60 <= delta <= 30*86400 + 60
    cx.close()


def test_grant_custom_days_when_specified(app_client_mem):
    client, app_module, db = app_client_mem
    r = client.post("/admin/membership/grant",
                    json={"email":"d90@example.com","source":"video","days":90},
                    headers=_ckey())
    assert r.status_code == 200
    cx = sqlite3.connect(db)
    granted_at, expires_at = cx.execute(
        "SELECT granted_at, expires_at FROM memberships WHERE email=?",
        ("d90@example.com",)
    ).fetchone()
    g = datetime.fromisoformat(granted_at.rstrip("Z"))
    e = datetime.fromisoformat(expires_at.rstrip("Z"))
    delta = (e - g).total_seconds()
    assert 90*86400 - 60 <= delta <= 90*86400 + 60
    cx.close()


def test_admin_escalations_requires_console_key(app_client_mem):
    client, _, _ = app_client_mem
    r = client.get("/admin/escalations")
    assert r.status_code in (401, 403)


def test_admin_escalations_returns_only_pending(app_client_mem):
    client, app_module, db = app_client_mem
    # Seed two rows: one pending, one replied
    import uuid
    now_iso = datetime.utcnow().isoformat()+"Z"
    cx = sqlite3.connect(db)
    cx.execute(
        "INSERT INTO escalation_queue "
        "(id, created_at, email, query_text, ai_response, ai_confidence, flag_reason, status) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), now_iso, "a@e.com", "pending q", "ai resp", 0.5, "member_request", "pending"))
    cx.execute(
        "INSERT INTO escalation_queue "
        "(id, created_at, email, query_text, ai_response, ai_confidence, flag_reason, status) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), now_iso, "b@e.com", "replied q", "ai resp", 0.5, "member_request", "replied"))
    cx.commit(); cx.close()
    r = client.get("/admin/escalations", headers=_ckey())
    assert r.status_code == 200
    arr = r.get_json()
    assert isinstance(arr, list)
    emails = [item["email"] for item in arr]
    assert "a@e.com" in emails
    assert "b@e.com" not in emails
