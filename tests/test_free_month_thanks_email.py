import os, sqlite3
from datetime import date, timedelta
import app as appmod
from dashboard import subscriptions as subs, referrals as rf, free_month as fm


M = "fm-thanks@example.com"
REFEREE = "fm-thanks-referee@example.com"


def _secret():
    return os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "test-secret")


def _member(cx, email, *, next_date, order_count=1):
    subs.init_subscriptions_table(cx); subs.migrate_add_membership_columns(cx); subs.migrate_add_free_months(cx)
    subs.migrate_add_term_cap_column(cx); subs.migrate_add_attribution_column(cx)
    subs.migrate_add_consent_column(cx); subs.migrate_add_failed_count(cx)
    cx.execute("DELETE FROM subscriptions WHERE email=?", (email,)); cx.commit()
    sid = subs.create_membership(cx, email=email, stripe_customer_id="cus", stripe_payment_method_id="pm",
                                 amount_cents=9900, next_charge_date=next_date, cadence_months=1,
                                 initial_order_count=order_count)
    cx.commit(); return sid


def test_comp_eligible_member_gets_thanks_not_charge_notice(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    monkeypatch.setenv("REFERRAL_FREE_MONTH_ENABLED", "1")
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _secret())
    sent = []
    monkeypatch.setattr(appmod, "_send_subscription_email",
                        lambda to, kind, data: sent.append((to, kind)) or ("smtp", None))
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session", lambda *a, **k: {"status": "succeeded", "id": "pi"})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1", "DisplayName": "x"})
    monkeypatch.setattr(appmod.qb, "create_invoice", lambda *a, **k: {"Id": "INV1"})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)

    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row; rf.init_tables(cx)
    # upcoming charge in 2 days => inside the 3-day heads-up window
    soon = (date.today() + timedelta(days=2)).isoformat()
    _member(cx, M, next_date=soon, order_count=1)
    _member(cx, REFEREE, next_date="2030-01-01", order_count=1)
    cx.execute("DELETE FROM referral_redemptions WHERE referee_email=?", (REFEREE,)); cx.commit()
    rf.record_redemption(cx, "CODE", M, REFEREE, "ref1")

    appmod.app.test_client().post("/api/cron/charge-subscriptions", headers={"X-Cron-Secret": _secret()})

    kinds_for_m = [k for (to, k) in sent if to == M]
    assert "free_month_thanks" in kinds_for_m
    assert "heads_up" not in kinds_for_m
    for e in (M, REFEREE):
        cx.execute("DELETE FROM subscriptions WHERE email=?", (e,))
    cx.execute("DELETE FROM referral_redemptions WHERE referee_email=?", (REFEREE,)); cx.commit(); cx.close()


M2 = "fm-churn@example.com"
REFEREE2 = "fm-churn-referee@example.com"


def _wipe_all(cx):
    # Both list_heads_up_due and list_due sweep the ENTIRE global subscriptions
    # table (no email filter), so leftover rows from other tests/fixtures would
    # otherwise be notified/charged/comped alongside this test's own rows.
    subs.init_subscriptions_table(cx)
    fm.init_comps_table(cx)
    cx.execute("DELETE FROM subscriptions")
    cx.execute("DELETE FROM membership_comps")
    cx.commit()


def test_referral_churn_after_headsup_does_not_charge(monkeypatch):
    """Two-phase regression test for the thanked-then-charged window.

    Phase 1 (heads-up, referral still active): member M is comp-eligible only
    via the live referral threshold. The heads-up pass must now BANK a free
    month immediately (not just re-check the referral later at charge time).

    Phase 2 (charge day, referral has churned): the referee no longer
    qualifies. Because the free month was banked at heads-up, Pass 2b's
    banked-check comps M regardless -- proving the "you will not be charged"
    promise is honored even though the live referral condition that earned it
    no longer holds by charge day.
    """
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    monkeypatch.setenv("REFERRAL_FREE_MONTH_ENABLED", "1")
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _secret())
    sent = []
    charges = []
    monkeypatch.setattr(appmod, "_send_subscription_email",
                        lambda to, kind, data: sent.append((to, kind)) or ("smtp", None))
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: charges.append(a[2]) or {"status": "succeeded", "id": "pi"})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1", "DisplayName": "x"})
    monkeypatch.setattr(appmod.qb, "create_invoice", lambda *a, **k: {"Id": "INV1"})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)

    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row; rf.init_tables(cx)
    _wipe_all(cx)

    # ── Phase 1: heads-up, referral still active ───────────────────────────
    soon = (date.today() + timedelta(days=2)).isoformat()   # inside 3-day window
    sid = _member(cx, M2, next_date=soon, order_count=1)
    _member(cx, REFEREE2, next_date="2030-01-01", order_count=1)  # active full referee, never due
    cx.execute("DELETE FROM referral_redemptions WHERE referee_email=?", (REFEREE2,)); cx.commit()
    rf.record_redemption(cx, "CODE2", M2, REFEREE2, "ref2")

    appmod.app.test_client().post("/api/cron/charge-subscriptions", headers={"X-Cron-Secret": _secret()})

    kinds_for_m = [k for (to, k) in sent if to == M2]
    assert "free_month_thanks" in kinds_for_m
    assert "heads_up" not in kinds_for_m
    s = subs.get(cx, sid)
    assert int(s["free_months_remaining"]) == 1   # banked at heads-up, not deferred to charge day
    assert 9900 not in charges

    # ── Phase 2: charge day, referral has churned (referee's redemption gone) ─
    sent.clear()
    cx.execute("DELETE FROM referral_redemptions WHERE owner_email=?", (M2,)); cx.commit()
    subs.set_next_charge_date(cx, sid, date.today().isoformat())
    cx.commit()

    appmod.app.test_client().post("/api/cron/charge-subscriptions", headers={"X-Cron-Secret": _secret()})

    assert 9900 not in charges                     # comped from the bank, NOT charged
    s = subs.get(cx, sid)
    assert int(s["free_months_remaining"]) == 0    # bank consumed by the comp

    for e in (M2, REFEREE2):
        cx.execute("DELETE FROM subscriptions WHERE email=?", (e,))
    cx.execute("DELETE FROM referral_redemptions WHERE referee_email=?", (REFEREE2,)); cx.commit(); cx.close()
