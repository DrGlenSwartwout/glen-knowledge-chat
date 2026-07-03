import sqlite3
from dashboard import analysis_quota as q

def _cx():
    cx = sqlite3.connect(":memory:"); q.init_analysis_quota_table(cx); return cx

def test_one_claim_per_calendar_month():
    cx = _cx()
    assert q.try_claim(cx, "a@b.com", month="2026-07") is True
    assert q.try_claim(cx, "a@b.com", month="2026-07") is False   # second in same month
    assert q.try_claim(cx, "a@b.com", month="2026-08") is True    # new month ok
    assert q.claimed_this_month(cx, "a@b.com", month="2026-08") is True
