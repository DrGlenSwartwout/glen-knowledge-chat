import sqlite3
from dashboard import sales_votes as sv

def _cx(): return sqlite3.connect(":memory:")

def test_record_pick_upsert_one_row_per_session_kind():
    cx = _cx()
    sv.record_pick(cx, "longevity", "botanical", 1, "sessA")
    sv.record_pick(cx, "longevity", "botanical", 2, "sessA")   # re-pick updates
    assert sv.get_picks(cx, "longevity", session_id="sessA")["botanical"] == 2
    assert sv.tally(cx, "longevity") == {"botanical": {2: 1}}   # one row, last choice

def test_picked_both_requires_real_pick_in_both():
    cx = _cx()
    sv.record_pick(cx, "longevity", "botanical", 1, "sessA")
    assert sv.picked_both(cx, "longevity", session_id="sessA") is False
    sv.record_pick(cx, "longevity", "mechanism", 0, "sessA")    # neither
    assert sv.picked_both(cx, "longevity", session_id="sessA") is False
    sv.record_pick(cx, "longevity", "mechanism", 1, "sessA")
    assert sv.picked_both(cx, "longevity", session_id="sessA") is True

def test_email_backfill_enables_match_by_email():
    cx = _cx()
    sv.record_pick(cx, "longevity", "botanical", 1, "sessA")            # anon
    sv.record_pick(cx, "longevity", "mechanism", 1, "sessA", "a@b.co")  # identified -> backfills
    assert sv.picked_both(cx, "longevity", email="a@b.co") is True      # both now carry the email

def test_tally_excludes_neither():
    cx = _cx()
    sv.record_pick(cx, "x", "botanical", 1, "s1")
    sv.record_pick(cx, "x", "botanical", 1, "s2")
    sv.record_pick(cx, "x", "botanical", 0, "s3")   # neither
    assert sv.tally(cx, "x") == {"botanical": {1: 2}}
