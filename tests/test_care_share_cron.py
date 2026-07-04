"""Wires Task 4's care_share.credit_for_charge into the renewal cron
(/api/cron/charge-subscriptions membership branch).

Mirrors the harness in test_membership_charge_cron.py: seeds against the real
LOG_DB, stubs Stripe/QBO/email/_ingest_order, posts to the cron endpoint with
the secret header. Seeds one ATTRIBUTED due membership and one UNATTRIBUTED
due membership, patches dashboard.care_share.credit_for_charge to a recorder,
and asserts it fires for the attributed one (with charge_cents == amount_cents)
using the POST-advance order_count, and also fires for the unattributed one
(Task 4's credit_for_charge itself no-ops when unattributed).
"""
import os
import sqlite3

import app as appmod
from dashboard import care_share
from dashboard import subscriptions as subs

TEST_EMAIL_ATTR = "cc-attr@example.com"
TEST_EMAIL_UNATTR = "cc-unattr@example.com"
PRACTITIONER_ID = "prac-turnkey-1"


def _cron_secret():
    return os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "test-secret")


def _headers():
    return {"X-Cron-Secret": _cron_secret()}


def _enable(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _cron_secret())


def _seed_membership(cx, *, email, amount_cents, attributed_practitioner_id=None,
                      next_charge_date="2000-01-01", initial_order_count=0):
    subs.init_subscriptions_table(cx)
    subs.migrate_add_failed_count(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    subs.migrate_add_attribution_column(cx)
    cx.execute("DELETE FROM subscriptions WHERE email=?", (email,))
    cx.commit()
    sid = subs.create_membership(
        cx,
        email=email,
        stripe_customer_id=f"cus_{email[:6]}",
        stripe_payment_method_id=f"pm_{email[:6]}",
        amount_cents=amount_cents,
        next_charge_date=next_charge_date,
        cadence_months=1,
        initial_order_count=initial_order_count,
        attributed_practitioner_id=attributed_practitioner_id,
    )
    cx.commit()
    return sid


def _stub_externals(monkeypatch):
    monkeypatch.setattr(
        appmod.stripe_pay, "charge_off_session",
        lambda *a, **k: {"status": "succeeded", "id": "pi_x"},
    )
    monkeypatch.setattr(appmod.qb, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1", "DisplayName": "x"})
    monkeypatch.setattr(appmod.qb, "create_invoice",
                        lambda *a, **k: {"Id": "INV1"})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))


def test_successful_membership_charge_credits_attributed_doctor(monkeypatch):
    _enable(monkeypatch)
    _stub_externals(monkeypatch)

    calls = []

    def _recorder(sub, *, charge_cents):
        calls.append({"sub_id": sub.get("id"), "order_count": sub.get("order_count"),
                       "attributed_practitioner_id": sub.get("attributed_practitioner_id"),
                       "charge_cents": charge_cents})
        return 0

    monkeypatch.setattr(care_share, "credit_for_charge", _recorder)

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    sid_attr = _seed_membership(
        cx, email=TEST_EMAIL_ATTR, amount_cents=9900,
        attributed_practitioner_id=PRACTITIONER_ID, initial_order_count=2,
    )
    sid_unattr = _seed_membership(
        cx, email=TEST_EMAIL_UNATTR, amount_cents=5000,
        attributed_practitioner_id=None,
    )

    c = appmod.app.test_client()
    r = c.post("/api/cron/charge-subscriptions", headers=_headers())
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["ok"] is True
    assert body["charged"] >= 2

    by_sid = {call["sub_id"]: call for call in calls}

    # Attributed membership: credit_for_charge invoked with the POST-advance
    # order_count (2 -> 3) and the full charge amount.
    assert sid_attr in by_sid, f"credit_for_charge was not invoked for attributed sub: {calls}"
    attr_call = by_sid[sid_attr]
    assert attr_call["charge_cents"] == 9900
    assert attr_call["order_count"] == 3
    assert attr_call["attributed_practitioner_id"] == PRACTITIONER_ID

    # Unattributed membership: credit_for_charge is still invoked (Task 4's
    # implementation no-ops on a missing attribution) but with no practitioner.
    assert sid_unattr in by_sid, f"credit_for_charge was not invoked for unattributed sub: {calls}"
    unattr_call = by_sid[sid_unattr]
    assert unattr_call["charge_cents"] == 5000
    assert not unattr_call["attributed_practitioner_id"]

    cx.execute("DELETE FROM subscriptions WHERE email IN (?, ?)",
               (TEST_EMAIL_ATTR, TEST_EMAIL_UNATTR))
    cx.commit()
    cx.close()


def test_failed_membership_charge_does_not_credit(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(
        appmod.stripe_pay, "charge_off_session",
        lambda *a, **k: {"status": "failed", "id": "pi_x", "decline_code": "generic_decline"},
    )
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))

    calls = []
    monkeypatch.setattr(
        care_share, "credit_for_charge",
        lambda sub, *, charge_cents: calls.append(sub.get("id")) or 0,
    )

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    sid = _seed_membership(
        cx, email=TEST_EMAIL_ATTR, amount_cents=9900,
        attributed_practitioner_id=PRACTITIONER_ID,
    )

    c = appmod.app.test_client()
    r = c.post("/api/cron/charge-subscriptions", headers=_headers())
    assert r.status_code == 200, r.data

    assert sid not in calls, "credit_for_charge must not fire on a failed charge"

    cx.execute("DELETE FROM subscriptions WHERE email=?", (TEST_EMAIL_ATTR,))
    cx.commit()
    cx.close()
