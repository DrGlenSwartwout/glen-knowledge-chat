import sqlite3
from dashboard import client_portal as cp
from dashboard.portal_provision import ensure_portal_for_buyer


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "t.db"))
    cp.init_client_portal_table(cx)
    return cx


def test_mints_token_first_time(tmp_path):
    cx = _cx(tmp_path)
    tok = ensure_portal_for_buyer(cx, "Buyer@X.com ", "Buyer")
    assert tok and isinstance(tok, str)


def test_idempotent_repeat_returns_none(tmp_path):
    cx = _cx(tmp_path)
    assert ensure_portal_for_buyer(cx, "b@x.com", "B")          # first mints
    assert ensure_portal_for_buyer(cx, "b@x.com", "B") is None  # repeat: no new token


def test_empty_email_returns_none(tmp_path):
    cx = _cx(tmp_path)
    assert ensure_portal_for_buyer(cx, "", "B") is None
    assert ensure_portal_for_buyer(cx, None, "B") is None


def test_mints_none_status(tmp_path):
    cx = _cx(tmp_path)
    ensure_portal_for_buyer(cx, "c@x.com", "C")
    rec = cp.get_portal_content_by_email(cx, "c@x.com")
    assert (rec.get("content") or {}).get("biofield_status") == "none"
