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
