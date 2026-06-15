"""Tests for the membership branch of /api/cron/charge-subscriptions
(Mechanic 1 — program-bundled live-group coaching).

Mirrors the harness in test_subscriptions_cron.py: seeds against the real
LOG_DB, stubs all external calls, posts to the cron endpoint with the secret
header. Here we seed ONE due kind='membership' subscription and assert it is
charged the flat amount_cents off-session and recorded as a membership order.
"""
import os
import sqlite3

import app as appmod
from dashboard import subscriptions as subs


def _cron_secret():
    return os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "test-secret")


def _headers():
    return {"X-Cron-Secret": _cron_secret()}


def _seed_membership(cx, *, amount_cents=9900, next_charge_date="2000-01-01"):
    """Insert one due membership subscription and return its id."""
    subs.init_subscriptions_table(cx)
    subs.migrate_add_failed_count(cx)
    subs.migrate_add_membership_columns(cx)
    cx.execute("DELETE FROM subscriptions WHERE email='mbr-test@example.com'")
    cx.commit()
    sid = subs.create_membership(
        cx,
        email="mbr-test@example.com",
        stripe_customer_id="cus_mbr",
        stripe_payment_method_id="pm_mbr",
        amount_cents=amount_cents,
        next_charge_date=next_charge_date,
        cadence_months=1,
    )
    cx.commit()
    return sid


def _enable(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _cron_secret())


def test_membership_charge_flat_amount_and_advances(monkeypatch):
    _enable(monkeypatch)

    charge_amounts = []
    monkeypatch.setattr(
        appmod.stripe_pay, "charge_off_session",
        lambda *a, **k: charge_amounts.append(a[2]) or {"status": "succeeded", "id": "pi_x"},
    )
    monkeypatch.setattr(appmod.qb, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1", "DisplayName": "x"})
    monkeypatch.setattr(appmod.qb, "create_invoice",
                        lambda *a, **k: {"Id": "INV1"})
    monkeypatch.setattr(appmod, "_send_subscription_email",
                        lambda *a, **k: ("smtp", None))

    ingested = []
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: ingested.append(kw))

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    sid = _seed_membership(cx, amount_cents=9900, next_charge_date="2000-01-01")

    c = appmod.app.test_client()
    r = c.post("/api/cron/charge-subscriptions", headers=_headers())
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["ok"] is True
    assert body["charged"] >= 1

    # Charged the flat amount, not a product-priced total
    assert 9900 in charge_amounts

    # Order recorded as a membership for the flat amount
    mbr = [o for o in ingested if o.get("source") == "membership"]
    assert len(mbr) == 1
    assert mbr[0]["total_cents"] == 9900

    # next_charge_date advanced by 1 month
    s = subs.get(cx, sid)
    assert s["next_charge_date"] == subs.add_months("2000-01-01", 1)

    cx.execute("DELETE FROM subscriptions WHERE email='mbr-test@example.com'"); cx.commit()
    cx.close()


def test_membership_dry_run_charges_nothing(monkeypatch):
    _enable(monkeypatch)

    charge_amounts = []
    monkeypatch.setattr(
        appmod.stripe_pay, "charge_off_session",
        lambda *a, **k: charge_amounts.append(a[2]) or {"status": "succeeded", "id": "pi_x"},
    )
    monkeypatch.setattr(appmod.qb, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1", "DisplayName": "x"})
    monkeypatch.setattr(appmod.qb, "create_invoice",
                        lambda *a, **k: {"Id": "INV1"})
    monkeypatch.setattr(appmod, "_send_subscription_email",
                        lambda *a, **k: ("smtp", None))

    ingested = []
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: ingested.append(kw))

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    sid = _seed_membership(cx, amount_cents=9900, next_charge_date="2000-01-01")

    c = appmod.app.test_client()
    r = c.post("/api/cron/charge-subscriptions?dry_run=1", headers=_headers())
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["ok"] is True
    assert body["dry_run"] is True
    assert body["charged"] >= 1  # dry preview counts the due membership

    # No charge, no order ingested, no advance
    assert charge_amounts == []
    assert ingested == []
    s = subs.get(cx, sid)
    assert s["next_charge_date"] == "2000-01-01"

    cx.execute("DELETE FROM subscriptions WHERE email='mbr-test@example.com'"); cx.commit()
    cx.close()
