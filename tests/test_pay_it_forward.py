import sqlite3
from dashboard import pay_it_forward as pif
from dashboard import points
from dashboard import referrals


def _cx():
    cx = sqlite3.connect(":memory:")
    points.init_points_table(cx)
    return cx


def test_award_milestone_credits_points():
    cx = _cx()
    pif.award_milestone(cx, "Member@X.com", milestone_key="program_complete_1")
    assert points.balance(cx, "member@x.com") == pif.MILESTONE_REWARD_CENTS


def test_award_milestone_idempotent_per_key():
    cx = _cx()
    pif.award_milestone(cx, "m@x.com", milestone_key="scan_improved_q1")
    pif.award_milestone(cx, "m@x.com", milestone_key="scan_improved_q1")
    assert points.balance(cx, "m@x.com") == pif.MILESTONE_REWARD_CENTS


def test_award_milestone_distinct_keys_stack():
    cx = _cx()
    pif.award_milestone(cx, "m@x.com", milestone_key="a")
    pif.award_milestone(cx, "m@x.com", milestone_key="b")
    assert points.balance(cx, "m@x.com") == pif.MILESTONE_REWARD_CENTS * 2


def test_award_milestone_ignores_blank():
    cx = _cx()
    pif.award_milestone(cx, "", milestone_key="x")
    pif.award_milestone(cx, "m@x.com", milestone_key="")
    assert points.balance(cx, "m@x.com") == 0


def _seed_redemption(cx, owner, referee):
    referrals.init_tables(cx)
    referrals.record_redemption(cx, "CODE", owner, referee, order_ref=f"o:{referee}")


def test_chain_summary_counts_two_levels():
    cx = _cx()
    # A gifted B and C (L1); B gifted D (L2)
    _seed_redemption(cx, "a@x.com", "b@x.com")
    _seed_redemption(cx, "a@x.com", "c@x.com")
    _seed_redemption(cx, "b@x.com", "d@x.com")
    s = pif.chain_summary(cx, "A@X.com")
    assert s["l1"] == 2
    assert s["l2"] == 1
    assert s["reached"] == 3
    assert s["levels"] == [2, 1]


def test_chain_summary_empty_for_unknown():
    cx = _cx()
    referrals.init_tables(cx)
    s = pif.chain_summary(cx, "nobody@x.com")
    assert s == {"reached": 0, "l1": 0, "l2": 0, "levels": []}


def test_chain_summary_excludes_self_and_dedupes():
    cx = _cx()
    _seed_redemption(cx, "a@x.com", "b@x.com")
    _seed_redemption(cx, "b@x.com", "a@x.com")  # cycle back to seed: must not recount A
    s = pif.chain_summary(cx, "a@x.com")
    assert s["reached"] == 1
    assert s["l1"] == 1
    assert s["l2"] == 0
    assert s["levels"] == [1, 0]
