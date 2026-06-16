"""create_magic_link_token honors a custom ttl_min (used by the emailed cert
portal invite so the link isn't dead on arrival), defaulting to the short
interactive-login TTL otherwise."""
import sqlite3
from datetime import datetime, timezone
from dashboard import practitioner_portal as pp


def _expiry(db, token):
    with sqlite3.connect(db) as cx:
        row = cx.execute(
            "SELECT created_at, expires_at FROM auth_tokens WHERE token_hash=?",
            (pp._hash(token),)).fetchone()
    c = datetime.fromisoformat(row[0].rstrip("Z"))
    e = datetime.fromisoformat(row[1].rstrip("Z"))
    return round((e - c).total_seconds() / 60)


def test_default_ttl_is_login_window(tmp_path):
    db = str(tmp_path / "t.db")
    tok = pp.create_magic_link_token("pid1", "a@x.com", db_path=db)
    assert _expiry(db, tok) == pp.MAGIC_TTL_MIN  # 15


def test_custom_ttl_for_invite(tmp_path):
    db = str(tmp_path / "t.db")
    tok = pp.create_magic_link_token("pid1", "a@x.com", ttl_min=7 * 24 * 60, db_path=db)
    assert _expiry(db, tok) == 7 * 24 * 60  # 10080 minutes = 7 days
