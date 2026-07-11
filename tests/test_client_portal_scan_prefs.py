import sqlite3
from dashboard import client_portal as cp

def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "chat_log.db"))
    cp.init_client_portal_table(cx)
    return cx

def test_auto_advance_defaults_true_and_toggles(tmp_path):
    cx = _cx(tmp_path)
    cp.upsert_portal(cx, "a@x.com", "A", {"biofield_status": "confirmed"})
    assert cp.get_auto_advance(cx, "a@x.com") is True          # absent = default on
    assert cp.set_auto_advance(cx, "a@x.com", False) is True
    assert cp.get_auto_advance(cx, "a@x.com") is False
    # unrelated content preserved
    assert cp.get_portal_content_by_email(cx, "a@x.com")["content"]["biofield_status"] == "confirmed"

def test_current_scan_set_and_get(tmp_path):
    cx = _cx(tmp_path)
    cp.upsert_portal(cx, "a@x.com", "A", {})
    assert cp.get_current_scan(cx, "a@x.com") is None
    assert cp.set_current_scan(cx, "a@x.com", "2026-07-09") is True
    assert cp.get_current_scan(cx, "a@x.com") == "2026-07-09"

def test_set_on_missing_portal_returns_false(tmp_path):
    cx = _cx(tmp_path)
    assert cp.set_auto_advance(cx, "nobody@x.com", False) is False
