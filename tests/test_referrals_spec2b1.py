import sqlite3
from dashboard import referrals as rf


def _cx():
    return sqlite3.connect(":memory:")


def test_get_or_create_code_stable_and_unique():
    cx = _cx()
    c1 = rf.get_or_create_code(cx, "Owner@X.com")
    c2 = rf.get_or_create_code(cx, "owner@x.com")          # same person (lowercased) -> same code
    assert c1 == c2 and c1
    c3 = rf.get_or_create_code(cx, "other@x.com")
    assert c3 != c1
    assert rf.owner_of(cx, c1) == "owner@x.com"
    assert rf.owner_of(cx, "NOPE") is None


def test_resolve_valid_and_guards():
    cx = _cx()
    code = rf.get_or_create_code(cx, "owner@x.com")
    # valid referee
    assert rf.resolve(cx, code, "friend@x.com", pct=10) == {"owner_email": "owner@x.com", "coupon_pct": 10}
    # self-referral blocked (case-insensitive)
    assert rf.resolve(cx, code, "OWNER@x.com", pct=10) is None
    # unknown code
    assert rf.resolve(cx, "NOPE", "friend@x.com", pct=10) is None


def test_one_redemption_per_referee_ever():
    cx = _cx()
    code = rf.get_or_create_code(cx, "owner@x.com")
    assert rf.has_redeemed(cx, "friend@x.com") is False
    assert rf.record_redemption(cx, code, "owner@x.com", "Friend@x.com", "INV-1") is True
    assert rf.has_redeemed(cx, "friend@x.com") is True       # lowercased
    # a second redemption by the same referee is a no-op insert, and resolve now blocks
    assert rf.record_redemption(cx, code, "owner@x.com", "friend@x.com", "INV-2") is False
    assert rf.resolve(cx, code, "friend@x.com", pct=10) is None
