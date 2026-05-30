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
