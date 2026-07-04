# tests/test_free_month_module.py
import sqlite3
import app as appmod
from dashboard import subscriptions as subs, referrals as rf, free_month as fm

REFERRER = "fm-referrer@example.com"
REFEREE = "fm-referee@example.com"


def _fresh(cx):
    subs.init_subscriptions_table(cx); subs.migrate_add_membership_columns(cx)
    subs.migrate_add_free_months(cx); rf.init_tables(cx); fm.init_comps_table(cx)
    for e in (REFERRER, REFEREE):
        cx.execute("DELETE FROM subscriptions WHERE email=?", (e,))
    cx.execute("DELETE FROM referral_redemptions WHERE owner_email=? OR referee_email=?", (REFERRER, REFEREE))
    cx.execute("DELETE FROM membership_comps WHERE email IN (?,?)", (REFERRER, REFEREE))
    cx.commit()


def _member(cx, email, *, order_count=1, skip_next=0):
    sid = subs.create_membership(
        cx, email=email, stripe_customer_id="cus", stripe_payment_method_id="pm",
        amount_cents=9900, next_charge_date="2026-07-01", cadence_months=1,
        initial_order_count=order_count)
    if skip_next:
        subs.set_skip_next(cx, sid, True)
    cx.commit()
    return sid


def test_grant_free_month_banks_and_is_idempotent():
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    _fresh(cx); sid = _member(cx, REFERRER)
    r1 = fm.grant_free_month(cx, REFERRER, reason="bounty", idem_key="k1")
    assert r1["free_months_remaining"] == 1
    r2 = fm.grant_free_month(cx, REFERRER, reason="bounty", idem_key="k1")  # replay
    assert r2["free_months_remaining"] == 1  # not 2
    cx.close()


def test_grant_free_month_none_without_membership():
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    _fresh(cx)
    assert fm.grant_free_month(cx, REFERRER, reason="bounty", idem_key="k2") is None
    cx.close()


def test_has_active_paying_referral_true_only_when_referee_is_full():
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    _fresh(cx); _member(cx, REFERRER)
    rf.record_redemption(cx, "CODE", REFERRER, REFEREE, "ref1")
    assert fm.has_active_paying_referral(cx, REFERRER) is False  # referee not a member yet
    _member(cx, REFEREE, order_count=1)
    assert fm.has_active_paying_referral(cx, REFERRER) is True
    cx.close()


def test_has_active_paying_referral_excludes_trial_and_paused():
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    _fresh(cx); _member(cx, REFERRER)
    rf.record_redemption(cx, "CODE", REFERRER, REFEREE, "ref1")
    _member(cx, REFEREE, order_count=0)  # trial (never charged)
    assert fm.has_active_paying_referral(cx, REFERRER) is False
    cx.execute("DELETE FROM subscriptions WHERE email=?", (REFEREE,)); cx.commit()
    _member(cx, REFEREE, order_count=1, skip_next=1)  # full but paused
    assert fm.has_active_paying_referral(cx, REFERRER) is False
    cx.close()


def test_comp_membership_cycle_from_bank_decrements_and_is_idempotent():
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    _fresh(cx); sid = _member(cx, REFERRER)
    fm.grant_free_month(cx, REFERRER, reason="bounty", idem_key="g1")
    ok = fm.comp_membership_cycle(cx, sid, reason="banked", idem_key="c1", from_bank=True)
    assert ok is True
    assert int(subs.get(cx, sid)["free_months_remaining"]) == 0
    assert subs.get(cx, sid)["next_charge_date"] == subs.add_months("2026-07-01", 1)
    # replay same idem_key: no second advance, no negative counter
    assert fm.comp_membership_cycle(cx, sid, reason="banked", idem_key="c1", from_bank=True) is False
    cx.close()


def test_comp_from_bank_false_when_no_banked_month():
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    _fresh(cx); sid = _member(cx, REFERRER)
    assert fm.comp_membership_cycle(cx, sid, reason="banked", idem_key="c2", from_bank=True) is False
    cx.close()
