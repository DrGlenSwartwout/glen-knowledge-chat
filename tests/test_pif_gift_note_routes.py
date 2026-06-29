import sqlite3
import app as appmod


def _db(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    # auth_tokens table (mirror the app's schema essentials)
    cx = sqlite3.connect(db)
    cx.execute("CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT PRIMARY KEY, email TEXT, "
               "purpose TEXT, extra TEXT, created_at TEXT, expires_at TEXT, consumed_at TEXT)")
    cx.commit(); cx.close()
    return db


def test_gift_note_token_roundtrip(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    tok = appmod._mint_gift_note_link("B@x.com", order_ref="o1")
    out = appmod._validate_gift_note_link(tok)
    assert out == {"email": "b@x.com", "order_ref": "o1"}


def test_gift_note_token_consumed_is_rejected(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    tok = appmod._mint_gift_note_link("b@x.com", order_ref="o1")
    appmod._consume_gift_note_token(tok)
    assert appmod._validate_gift_note_link(tok) is None


def test_gift_note_token_bad_is_none(monkeypatch, tmp_path):
    _db(monkeypatch, tmp_path)
    assert appmod._validate_gift_note_link("not-a-real-token") is None


def test_gift_note_token_rejects_other_purpose(monkeypatch, tmp_path):
    """Prove a token minted under a different purpose cannot validate as a gift-note token."""
    import json
    _db(monkeypatch, tmp_path)
    # mint a token row under a different purpose with the SAME hashing scheme
    plain = "someplaintexttoken123"
    th = appmod._hash_token(plain)
    cx = sqlite3.connect(appmod.LOG_DB)
    cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
               "VALUES (?,?,?,?,?,?)",
               (th, "x@x.com", "membership_magic_link", json.dumps({"order_ref": "o1"}),
                "2026-01-01T00:00:00Z", "2099-01-01T00:00:00Z"))
    cx.commit(); cx.close()
    assert appmod._validate_gift_note_link(plain) is None   # wrong purpose -> rejected


def test_gift_note_token_expired_is_none(monkeypatch, tmp_path):
    """A token minted with ttl_min=0 (already expired) must validate to None."""
    _db(monkeypatch, tmp_path)
    tok = appmod._mint_gift_note_link("b@x.com", order_ref="o1", ttl_min=0)
    assert appmod._validate_gift_note_link(tok) is None
