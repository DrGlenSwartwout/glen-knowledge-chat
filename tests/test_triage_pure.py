import sqlite3
from datetime import datetime
from dashboard import triage

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    triage.init_triage_tables(cx); return cx

def test_invite_roundtrip_and_mark_booked():
    cx = _cx()
    tok = triage.create_invite(cx, "P@x.com", "Pat", "glen")
    inv = triage.resolve_invite(cx, tok)
    assert inv and inv["email"] == "p@x.com" and inv["practitioner"] == "glen"
    assert inv["status"] == "invited" and inv["booked_start"] is None
    triage.mark_booked(cx, tok, "2026-07-06T13:00:00")
    inv2 = triage.resolve_invite(cx, tok)
    assert inv2["status"] == "booked" and inv2["booked_start"] == "2026-07-06T13:00:00"

def test_resolve_bad_and_expired():
    cx = _cx()
    assert triage.resolve_invite(cx, "nope") is None
    now = datetime(2026, 7, 1, 12, 0)
    tok = triage.create_invite(cx, "e@x.com", "E", "rae", days=7, _now=now)
    assert triage.resolve_invite(cx, tok, _now=datetime(2026, 7, 6)) is not None   # day 5 ok
    assert triage.resolve_invite(cx, tok, _now=datetime(2026, 7, 9)) is None        # day 8 expired
