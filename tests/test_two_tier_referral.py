# tests/test_two_tier_referral.py
"""Two-tier referral points: owner_of_referee lineage, Tier-2 settlement in
_settle_referrer_reward (half the Tier-1 rate, non-cashable, forward-only,
cycle-guarded, idempotent), and the per-tier earnings in /api/pif/summary."""
import sqlite3
import app as appmod
from dashboard import referrals as rf, points as pts


def _cx():
    cx = sqlite3.connect(":memory:")
    rf.init_tables(cx)
    pts.init_points_table(cx)
    return cx


def _chain_CB_BA(cx):
    # C referred by B (order ordC); B referred by A. So for C's purchase: L1=B, L2=A.
    rf.record_redemption(cx, "C1", "b@x.com", "c@x.com", "ordC")
    rf.record_redemption(cx, "C2", "a@x.com", "b@x.com", "")


def _on(monkeypatch, pct=10, tier2=True):
    monkeypatch.setattr(appmod, "_REFERRALS", True)
    monkeypatch.setattr(appmod, "REFERRAL_TIER2_ENABLED", tier2)
    monkeypatch.setattr(appmod, "_referrer_reward_pct", lambda: pct)


# -- unit: lineage + earnings helpers ---------------------------------------

def test_owner_of_referee():
    cx = _cx()
    rf.record_redemption(cx, "C", "a@x.com", "b@x.com", "")
    assert rf.owner_of_referee(cx, "b@x.com") == "a@x.com"
    assert rf.owner_of_referee(cx, "nobody@x.com") == ""


def test_earned_by_reason():
    cx = _cx()
    pts.credit(cx, "a@x.com", value_cents=500, reason="referral_reward_l2", order_ref="r1")
    pts.credit(cx, "a@x.com", value_cents=300, reason="referral_reward_l2", order_ref="r2")
    pts.credit(cx, "a@x.com", value_cents=100, reason="referral_reward", order_ref="r3")
    assert pts.earned_by_reason(cx, "a@x.com", "referral_reward_l2") == 800
    assert pts.earned_by_reason(cx, "a@x.com", "referral_reward") == 100
    assert pts.earned_by_reason(cx, "a@x.com", "nope") == 0


# -- settlement -------------------------------------------------------------

def test_tier2_credits_l2_half(monkeypatch):
    cx = _cx(); _chain_CB_BA(cx); _on(monkeypatch, pct=10)
    appmod._settle_referrer_reward(cx, {"total_cents": 10000, "shipping_cents": 0}, "ordC")
    assert pts.balance(cx, "b@x.com") == 1000   # Tier-1 = 10% of 10000
    assert pts.balance(cx, "a@x.com") == 500     # Tier-2 = half = 5%
    assert pts.earned_by_reason(cx, "a@x.com", "referral_reward_l2") == 500


def test_tier2_flag_off_no_l2(monkeypatch):
    cx = _cx(); _chain_CB_BA(cx); _on(monkeypatch, pct=10, tier2=False)
    appmod._settle_referrer_reward(cx, {"total_cents": 10000}, "ordC")
    assert pts.balance(cx, "b@x.com") == 1000    # Tier-1 unchanged
    assert pts.balance(cx, "a@x.com") == 0        # no Tier-2 when flag off


def test_no_l2_when_l1_has_no_referrer(monkeypatch):
    cx = _cx()
    rf.record_redemption(cx, "C1", "b@x.com", "c@x.com", "ordC")  # B has no referrer row
    _on(monkeypatch, pct=10)
    appmod._settle_referrer_reward(cx, {"total_cents": 10000}, "ordC")
    assert pts.balance(cx, "b@x.com") == 1000
    assert pts.has_entry(cx, order_ref="referral_l2:c@x.com", reason="referral_reward_l2") is False


def test_no_self_dealing_cycle(monkeypatch):
    cx = _cx()
    rf.record_redemption(cx, "C1", "b@x.com", "c@x.com", "ordC")  # C referred by B
    rf.record_redemption(cx, "C2", "c@x.com", "b@x.com", "")       # B referred by C (cycle)
    _on(monkeypatch, pct=10)
    appmod._settle_referrer_reward(cx, {"total_cents": 10000}, "ordC")
    # L2 = owner_of(B) = C = the buyer -> guarded, no self-credit
    assert pts.balance(cx, "c@x.com") == 0
    assert pts.earned_by_reason(cx, "c@x.com", "referral_reward_l2") == 0


def test_tier2_idempotent_on_replay(monkeypatch):
    cx = _cx(); _chain_CB_BA(cx); _on(monkeypatch, pct=10)
    order = {"total_cents": 10000}
    appmod._settle_referrer_reward(cx, order, "ordC")
    appmod._settle_referrer_reward(cx, order, "ordC")  # replay -> rewarded_at early-return
    assert pts.balance(cx, "a@x.com") == 500   # exactly one Tier-2 credit
    assert pts.balance(cx, "b@x.com") == 1000


# -- summary ----------------------------------------------------------------

def test_pif_summary_per_tier(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod, "PAY_IT_FORWARD_ENABLED", True)
    monkeypatch.setattr(appmod, "is_member", lambda sid, email: True)
    with sqlite3.connect(db) as cx:
        rf.init_tables(cx)
        pts.init_points_table(cx)
        pts.credit(cx, "m@x.com", value_cents=700, reason="referral_reward", order_ref="r1")
        pts.credit(cx, "m@x.com", value_cents=250, reason="referral_reward_l2", order_ref="r2")
    body = appmod.app.test_client().get("/api/pif/summary?email=m@x.com").get_json()
    assert body["ok"] is True
    assert body["tier1_earned_cents"] == 700
    assert body["tier2_earned_cents"] == 250
