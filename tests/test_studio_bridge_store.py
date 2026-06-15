import sqlite3
from dashboard import studio_bridge as sb

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    sb.init_table(cx); return cx

def test_record_pending_and_get():
    cx = _cx()
    sb.record_pending(cx, "p@x.com", signup_via="self-attest")
    r = sb.get(cx, "p@x.com")
    assert r["status"] == "pending" and r["signup_via"] == "self-attest"
    assert sb.already_granted(cx, "p@x.com") is False

def test_record_pending_idempotent():
    cx = _cx()
    sb.record_pending(cx, "p@x.com", signup_via="self-attest")
    sb.record_pending(cx, "p@x.com", signup_via="receipt")  # same row; updates signup_via
    rows = cx.execute("SELECT COUNT(*) FROM studio_bridge_claims WHERE email=?", ("p@x.com",)).fetchone()[0]
    assert rows == 1
    assert sb.get(cx, "p@x.com")["signup_via"] == "receipt"

def test_mark_granted():
    cx = _cx()
    sb.record_pending(cx, "p@x.com", signup_via="receipt")
    sb.mark_granted(cx, "p@x.com", 42)
    r = sb.get(cx, "p@x.com")
    assert r["status"] == "granted" and r["sub_id"] == 42 and r["granted_at"]
    assert sb.already_granted(cx, "p@x.com") is True

def test_already_granted_unknown_email():
    cx = _cx()
    assert sb.already_granted(cx, "nobody@x.com") is False

def test_get_missing_returns_none():
    cx = _cx()
    assert sb.get(cx, "nobody@x.com") is None
