import sqlite3
import dashboard.practitioner_portal as pp_mod
from dashboard import dispensary_rewards as dr, referrals as rf, points


def _cx(monkeypatch):
    monkeypatch.setenv("REFERRALS", "true")
    monkeypatch.setenv("REFERRAL_TIER2_ENABLED", "true")
    monkeypatch.setenv("REFERRER_REWARD_PCT", "20")
    cx = sqlite3.connect(":memory:")
    rf.init_tables(cx)
    points.init_points_table(cx)
    return cx


def _order(pid="prac-1", patient="pat@x.com", total=7000, shipping=1300, ref="INV1"):
    return {"practitioner_id": pid, "email": patient, "total_cents": total,
            "shipping_cents": shipping, "get_cents": 0, "external_ref": ref}


def test_credits_upline_l2_no_l1(monkeypatch):
    cx = _cx(monkeypatch)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")  # who referred the doc
    got = dr.settle_dispensary_l2(cx, _order(), "INV1")
    # product 5700; L2 = 5700 * 20 // 200 = 570
    assert got == 570
    assert points.balance(cx, "upline@x.com") == 570
    assert points.balance(cx, "doc@x.com") == 0        # practitioner (L1) never credited
    assert points.balance(cx, "pat@x.com") == 0


def test_idempotent_per_invoice(monkeypatch):
    cx = _cx(monkeypatch)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
    dr.settle_dispensary_l2(cx, _order(), "INV1")
    assert dr.settle_dispensary_l2(cx, _order(), "INV1") == 0   # replay
    assert points.balance(cx, "upline@x.com") == 570


def test_reorder_new_invoice_pays_again(monkeypatch):
    cx = _cx(monkeypatch)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
    dr.settle_dispensary_l2(cx, _order(ref="INV1"), "INV1")
    dr.settle_dispensary_l2(cx, _order(ref="INV2"), "INV2")   # reorder
    assert points.balance(cx, "upline@x.com") == 1140          # 570 * 2


def test_no_l2_when_practitioner_has_no_upline(monkeypatch):
    cx = _cx(monkeypatch)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    assert dr.settle_dispensary_l2(cx, _order(), "INV1") == 0   # no redemption row for doc


def test_resolves_from_order_pid_not_patient_referral(monkeypatch):
    """The patient carries an Ambassador referral row; L2 must NOT follow that chain."""
    cx = _cx(monkeypatch)
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    rf.record_redemption(cx, "AMB", "ambassador@x.com", "pat@x.com", "INV-AMB")  # patient's own referrer
    rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
    dr.settle_dispensary_l2(cx, _order(), "INV1")
    assert points.balance(cx, "upline@x.com") == 570              # doc's upline, correct
    assert points.balance(cx, "ambassador@x.com") == 0            # not the patient's ambassador chain


def test_no_l2_when_tier2_off(monkeypatch):
    cx = _cx(monkeypatch)
    monkeypatch.setenv("REFERRAL_TIER2_ENABLED", "false")
    monkeypatch.setattr(pp_mod, "practitioner_email_by_id", lambda pid: "doc@x.com")
    rf.record_redemption(cx, "UP", "upline@x.com", "doc@x.com", "INV-DOC")
    assert dr.settle_dispensary_l2(cx, _order(), "INV1") == 0
