import sqlite3
from dashboard import pay_it_forward as pif
from dashboard import points


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
