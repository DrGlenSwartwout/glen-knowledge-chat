# tests/test_free_month_cron.py
import os, sqlite3
import app as appmod
from dashboard import subscriptions as subs, referrals as rf, free_month as fm

M = "fm-cron-member@example.com"
REFEREE = "fm-cron-referee@example.com"


def _secret():
    return os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "test-secret")


def _enable(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _secret())


def _stub(monkeypatch, charges):
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: charges.append(a[2]) or {"status": "succeeded", "id": "pi_x"})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1", "DisplayName": "x"})
    monkeypatch.setattr(appmod.qb, "create_invoice", lambda *a, **k: {"Id": "INV1"})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))


def _member(cx, email, *, next_date, order_count=1):
    subs.init_subscriptions_table(cx); subs.migrate_add_membership_columns(cx); subs.migrate_add_free_months(cx)
    cx.execute("DELETE FROM subscriptions WHERE email=?", (email,)); cx.commit()
    sid = subs.create_membership(cx, email=email, stripe_customer_id="cus", stripe_payment_method_id="pm",
                                 amount_cents=9900, next_charge_date=next_date, cadence_months=1,
                                 initial_order_count=order_count)
    cx.commit(); return sid


def _post():
    return appmod.app.test_client().post("/api/cron/charge-subscriptions", headers={"X-Cron-Secret": _secret()})


def test_banked_free_month_comps_instead_of_charging(monkeypatch):
    _enable(monkeypatch); charges = []; _stub(monkeypatch, charges)
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    fm.init_comps_table(cx); cx.execute("DELETE FROM membership_comps WHERE email=?", (M,)); cx.commit()
    sid = _member(cx, M, next_date="2000-01-01", order_count=1)
    fm.grant_free_month(cx, M, reason="bounty", idem_key=f"seed:{sid}")
    r = _post(); body = r.get_json()
    assert body["comped"] >= 1
    assert 9900 not in charges                      # not charged
    s = subs.get(cx, sid)
    assert s["order_count"] == 1                     # unchanged
    assert s["next_charge_date"] == subs.add_months("2000-01-01", 1)
    assert int(s["free_months_remaining"]) == 0      # bank consumed
    cx.execute("DELETE FROM subscriptions WHERE email=?", (M,)); cx.commit(); cx.close()


def test_referral_threshold_comps_when_flag_on(monkeypatch):
    _enable(monkeypatch); monkeypatch.setenv("REFERRAL_FREE_MONTH_ENABLED", "1")
    charges = []; _stub(monkeypatch, charges)
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    rf.init_tables(cx)
    sid = _member(cx, M, next_date="2000-01-01", order_count=1)
    _member(cx, REFEREE, next_date="2030-01-01", order_count=1)   # a full active referee
    cx.execute("DELETE FROM referral_redemptions WHERE referee_email=?", (REFEREE,)); cx.commit()
    rf.record_redemption(cx, "CODE", M, REFEREE, "ref1")
    r = _post(); assert r.get_json()["comped"] >= 1
    assert 9900 not in charges
    for e in (M, REFEREE):
        cx.execute("DELETE FROM subscriptions WHERE email=?", (e,))
    cx.execute("DELETE FROM referral_redemptions WHERE referee_email=?", (REFEREE,)); cx.commit(); cx.close()


def test_no_referral_charges_normally_flag_on(monkeypatch):
    _enable(monkeypatch); monkeypatch.setenv("REFERRAL_FREE_MONTH_ENABLED", "1")
    charges = []; _stub(monkeypatch, charges)
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    sid = _member(cx, M, next_date="2000-01-01", order_count=1)
    cx.execute("DELETE FROM referral_redemptions WHERE owner_email=?", (M,)); cx.commit()
    _post()
    assert 9900 in charges                            # charged as normal
    s = subs.get(cx, sid); assert s["order_count"] == 2
    cx.execute("DELETE FROM subscriptions WHERE email=?", (M,)); cx.commit(); cx.close()


def test_flag_off_no_bank_charges_normally(monkeypatch):
    _enable(monkeypatch); monkeypatch.delenv("REFERRAL_FREE_MONTH_ENABLED", raising=False)
    charges = []; _stub(monkeypatch, charges)
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    sid = _member(cx, M, next_date="2000-01-01", order_count=1)
    _member(cx, REFEREE, next_date="2030-01-01", order_count=1)
    cx.execute("DELETE FROM referral_redemptions WHERE referee_email=?", (REFEREE,)); cx.commit()
    rf.record_redemption(cx, "CODE", M, REFEREE, "ref1")   # would qualify if flag were on
    _post()
    assert 9900 in charges                            # flag off => charged
    for e in (M, REFEREE):
        cx.execute("DELETE FROM subscriptions WHERE email=?", (e,))
    cx.execute("DELETE FROM referral_redemptions WHERE referee_email=?", (REFEREE,)); cx.commit(); cx.close()
