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


def test_release_frees_the_claim_for_reuse_same_month():
    cx = _cx()
    assert q.try_claim(cx, "a@b.com", month="2026-07") is True
    assert q.claimed_this_month(cx, "a@b.com", month="2026-07") is True
    q.release(cx, "a@b.com", month="2026-07")
    assert q.claimed_this_month(cx, "a@b.com", month="2026-07") is False
    assert q.try_claim(cx, "a@b.com", month="2026-07") is True  # can claim again


def test_release_is_a_noop_when_no_claim_exists():
    cx = _cx()
    q.release(cx, "nobody@b.com", month="2026-07")  # must not raise
    assert q.claimed_this_month(cx, "nobody@b.com", month="2026-07") is False


def test_release_only_affects_the_given_month():
    cx = _cx()
    assert q.try_claim(cx, "a@b.com", month="2026-07") is True
    assert q.try_claim(cx, "a@b.com", month="2026-08") is True
    q.release(cx, "a@b.com", month="2026-07")
    assert q.claimed_this_month(cx, "a@b.com", month="2026-07") is False
    assert q.claimed_this_month(cx, "a@b.com", month="2026-08") is True  # untouched
