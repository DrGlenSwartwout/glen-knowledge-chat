"""Tests for term-cap enforcement in the membership branch of
/api/cron/charge-subscriptions (Task 2 — Continuous Care monthly).

A fixed-term Continuous Care membership sub (term_charges_total = 6 or 12)
must stop auto-charging once it has been charged its committed number of
times, instead of renewing forever like a legacy (NULL cap) membership.

Mirrors the harness in test_membership_charge_cron.py: seeds against the
real LOG_DB, stubs all external calls, posts to the cron endpoint with the
secret header.
"""
import os
import sqlite3

import app as appmod
from dashboard import subscriptions as subs

TEST_EMAIL_CAPPED = "term-cap-capped@example.com"
TEST_EMAIL_UNCAPPED = "term-cap-uncapped@example.com"


def _cron_secret():
    return os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "test-secret")


def _headers():
    return {"X-Cron-Secret": _cron_secret()}


def _enable(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _cron_secret())


def _seed(cx, email, *, cap, order_count, next_date="2000-01-01", amount_cents=9900):
    """Insert one due membership subscription near/at its term cap."""
    subs.init_subscriptions_table(cx)
    subs.migrate_add_failed_count(cx)
    subs.migrate_add_membership_columns(cx)
    subs.migrate_add_term_cap_column(cx)
    cx.execute("DELETE FROM subscriptions WHERE email=?", (email,))
    cx.commit()
    sid = subs.create_membership(
        cx,
        email=email,
        stripe_customer_id="cus_termcap",
        stripe_payment_method_id="pm_termcap",
        amount_cents=amount_cents,
        next_charge_date=next_date,
        cadence_months=1,
        term_charges_total=cap,
        initial_order_count=order_count,
    )
    cx.commit()
    return sid


def _stub_externals(monkeypatch, charge_id="ch_1"):
    monkeypatch.setattr(
        appmod.stripe_pay, "charge_off_session",
        lambda *a, **k: {"status": "succeeded", "id": charge_id},
    )
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice", lambda *a, **k: {"Id": "INV1"})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **k: None)
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: None)


def test_capped_sub_cancels_at_cap(monkeypatch):
    _enable(monkeypatch)
    _stub_externals(monkeypatch, charge_id="ch_1")

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    try:
        # cap=6, already charged 5 times -> this charge is #6, the term-completing one
        sid = _seed(cx, TEST_EMAIL_CAPPED, cap=6, order_count=5)

        c = appmod.app.test_client()
        r = c.post("/api/cron/charge-subscriptions", headers=_headers())
        assert r.status_code == 200, r.data
        body = r.get_json()
        assert body["ok"] is True
        assert body["charged"] >= 1

        row = subs.get(cx, sid)
        assert row["status"] == "cancelled"  # order_count hit 6 -> term over
    finally:
        cx.execute("DELETE FROM subscriptions WHERE email=?", (TEST_EMAIL_CAPPED,))
        cx.commit()
        cx.close()


def test_uncapped_sub_stays_active(monkeypatch):
    _enable(monkeypatch)
    _stub_externals(monkeypatch, charge_id="ch_2")

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    try:
        sid = _seed(cx, TEST_EMAIL_UNCAPPED, cap=None, order_count=99)

        c = appmod.app.test_client()
        r = c.post("/api/cron/charge-subscriptions", headers=_headers())
        assert r.status_code == 200, r.data

        row = subs.get(cx, sid)
        assert row["status"] == "active"
    finally:
        cx.execute("DELETE FROM subscriptions WHERE email=?", (TEST_EMAIL_UNCAPPED,))
        cx.commit()
        cx.close()
