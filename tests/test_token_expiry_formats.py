"""A live token must validate no matter which timestamp shape it was stored in.

auth_tokens.expires_at holds three shapes, depending on which minter wrote the
row (see dashboard/timeutil.py). Validators used to parse the way their own
minter happened to write, so a row written by a *different* minter raised
`can't compare offset-naive and offset-aware datetimes` inside a bare `except`
and was reported EXPIRED.

Each test below mints a token that expires an hour from now and asserts the
validator accepts it. On the old code the mismatched shapes fail.
"""
import importlib
import secrets
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

repo = Path(__file__).resolve().parent.parent
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CONSOLE_SECRET", raising=False)
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    return appmod


# The three shapes, each expressed as a function of a UTC instant.
SHAPES = {
    "offset_aware": lambda d: d.replace(tzinfo=timezone.utc).isoformat(),
    "z_suffixed":   lambda d: d.isoformat() + "Z",
    "bare_naive":   lambda d: d.isoformat(),
}


def _insert(appmod, purpose, shape, *, email="fmt@example.com", extra=None,
            delta=timedelta(hours=1)):
    """Insert a token expiring `delta` from now, stored in `shape`."""
    tok = secrets.token_urlsafe(32)
    naive_now = datetime.utcnow()
    fmt = SHAPES[shape]
    with sqlite3.connect(appmod.LOG_DB) as cx:
        cx.execute(
            "CREATE TABLE IF NOT EXISTS auth_tokens (token_hash TEXT PRIMARY KEY, "
            "email TEXT, purpose TEXT NOT NULL, extra TEXT, created_at TEXT NOT NULL, "
            "expires_at TEXT NOT NULL, consumed_at TEXT)")
        cx.execute(
            "INSERT INTO auth_tokens (token_hash, email, purpose, extra, created_at, expires_at) "
            "VALUES (?,?,?,?,?,?)",
            (appmod._hash_token(tok), email, purpose, extra,
             fmt(naive_now), fmt(naive_now + delta)))
        cx.commit()
    return tok


@pytest.mark.parametrize("shape", list(SHAPES))
def test_membership_magic_link_accepts_every_shape(tmp_path, monkeypatch, shape):
    appmod = _app(tmp_path, monkeypatch)
    tok = _insert(appmod, "membership_magic_link", shape)
    assert appmod._validate_membership_magic_link(tok) == "fmt@example.com"


@pytest.mark.parametrize("shape", list(SHAPES))
def test_lead_magnet_guide_accepts_every_shape(tmp_path, monkeypatch, shape):
    appmod = _app(tmp_path, monkeypatch)
    tok = _insert(appmod, "lead_magnet_guide", shape)
    assert appmod._validate_lead_magnet_guide_link(tok) == "fmt@example.com"


@pytest.mark.parametrize("shape", list(SHAPES))
def test_gift_note_accepts_every_shape(tmp_path, monkeypatch, shape):
    appmod = _app(tmp_path, monkeypatch)
    tok = _insert(appmod, "pif_gift_note", shape, extra='{"order_ref": "o-1"}')
    got = appmod._validate_gift_note_link(tok)
    assert got is not None and got["order_ref"] == "o-1"


@pytest.mark.parametrize("shape", list(SHAPES))
def test_expired_token_is_still_rejected_in_every_shape(tmp_path, monkeypatch, shape):
    """Tolerant parsing must not accidentally resurrect an expired token."""
    appmod = _app(tmp_path, monkeypatch)
    tok = _insert(appmod, "membership_magic_link", shape, delta=timedelta(hours=-1))
    assert appmod._validate_membership_magic_link(tok) is None


@pytest.mark.parametrize("shape", list(SHAPES))
def test_practitioner_invoice_token_accepts_every_shape(tmp_path, monkeypatch, shape):
    appmod = _app(tmp_path, monkeypatch)
    import json
    from dashboard import practitioner_portal as PP
    tok = _insert(appmod, "order_invoice", shape, extra=json.dumps({"order_id": "77"}))
    assert PP.order_id_from_invoice_token(tok, db_path=appmod.LOG_DB) == "77"


def test_invoice_validator_accepts_an_aware_now(tmp_path, monkeypatch):
    """Passing a tz-aware `now` used to raise inside a bare except and read as
    expired -- which is why test_invoice_token_expires passed for a wrong reason."""
    appmod = _app(tmp_path, monkeypatch)
    from dashboard import practitioner_portal as PP
    tok = PP.create_order_invoice_token(88, ttl_days=30, db_path=appmod.LOG_DB)
    soon = datetime.now(timezone.utc) + timedelta(days=1)
    assert PP.order_id_from_invoice_token(tok, now=soon, db_path=appmod.LOG_DB) == "88"
    later = datetime.now(timezone.utc) + timedelta(days=31)
    assert PP.order_id_from_invoice_token(tok, now=later, db_path=appmod.LOG_DB) is None
