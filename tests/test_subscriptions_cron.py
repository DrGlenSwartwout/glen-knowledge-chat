"""Tests for /api/cron/charge-subscriptions (Task 4 — daily charge scheduler).

Isolation strategy:
  - appmod.LOG_DB points at the REAL path (set by Doppler DATA_DIR at test
    startup); we connect to the same file and seed / assert against it.
  - All external calls (stripe_pay, qb, _price_cart, _send_subscription_email,
    _ingest_order) are monkeypatched so no real charges, invoices, or emails
    are made.
  - Each test creates a fresh subscription row with a past next_charge_date so
    the scheduler will pick it up.
"""
import os
import sqlite3

import pytest

import app as appmod
from dashboard import subscriptions as subs


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cron_secret():
    """Return the cron secret that the endpoint accepts."""
    return os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "test-secret")


def _seed_sub(cx, *, next_charge_date="2000-01-01", skip_next=False):
    """Insert an active subscription due on next_charge_date and return its id."""
    subs.init_subscriptions_table(cx)
    subs.migrate_add_failed_count(cx)
    sid = subs.create(
        cx,
        email="sub-test@example.com",
        stripe_customer_id="cus_test",
        stripe_payment_method_id="pm_test",
        items=[{"slug": "x", "qty": 1}],
        cadence_months=1,
        ship_address={"state": "CA"},
        next_charge_date=next_charge_date,
    )
    if skip_next:
        subs.set_skip_next(cx, sid, True)
    cx.commit()
    return sid


def _mock_price_cart(monkeypatch):
    monkeypatch.setattr(
        appmod, "_price_cart",
        lambda cart, **k: {
            "priced": {"total_cents": 5000, "get_cents": 0},
            "qbo_lines": [{"name": "X", "amount": 50.0, "qty": 1}],
            "items_rec": [{"name": "X", "qty": 1, "desc": "X"}],
            "discount_cents": 0,
            "points_redeemed_cents": 0,
            "shipping_cents": 0,
        },
    )


def _mock_qb(monkeypatch):
    monkeypatch.setattr(appmod.qb, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice",
                        lambda *a, **k: {"Id": "INV1", "TotalAmt": 50.0})


def _mock_ingest(monkeypatch):
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)


def _mock_email(monkeypatch):
    monkeypatch.setattr(appmod, "_send_subscription_email",
                        lambda *a, **k: ("smtp", None))


def _headers():
    return {"X-Cron-Secret": _cron_secret()}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_cron_charges_due_subscription_and_advances(monkeypatch):
    """A due active subscription is charged and advanced after success."""
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    # Allow the secret check to pass even if CRON_SECRET is unset in test env
    _secret = _cron_secret()
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _secret)

    charged_calls = []
    monkeypatch.setattr(
        appmod.stripe_pay, "charge_off_session",
        lambda *a, **k: charged_calls.append(a) or {"id": "pi_ok", "status": "succeeded"},
    )
    _mock_price_cart(monkeypatch)
    _mock_qb(monkeypatch)
    _mock_ingest(monkeypatch)
    _mock_email(monkeypatch)

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    sid = _seed_sub(cx, next_charge_date="2000-01-01")

    c = appmod.app.test_client()
    r = c.post("/api/cron/charge-subscriptions", headers=_headers())
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["ok"] is True
    assert body["charged"] >= 1

    # Sub must be advanced (order_count 0 -> 1)
    s = subs.get(cx, sid)
    assert s["order_count"] == 1, "advance_after_charge was not called"

    # failed_count must be 0 (reset after success)
    assert s.get("failed_count", 0) == 0

    # stripe_pay.charge_off_session must have been called at least once
    assert len(charged_calls) >= 1

    cx.close()


def test_dry_run_does_not_charge_or_advance(monkeypatch):
    """?dry_run=1 must not call stripe or mutate the subscription."""
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    _secret = _cron_secret()
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _secret)

    charged_calls = []
    monkeypatch.setattr(
        appmod.stripe_pay, "charge_off_session",
        lambda *a, **k: charged_calls.append(a) or {"id": "pi_dry", "status": "succeeded"},
    )
    _mock_price_cart(monkeypatch)
    _mock_qb(monkeypatch)
    _mock_ingest(monkeypatch)
    _mock_email(monkeypatch)

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    sid = _seed_sub(cx, next_charge_date="2000-01-01")

    c = appmod.app.test_client()
    r = c.post("/api/cron/charge-subscriptions?dry_run=1", headers=_headers())
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["ok"] is True
    assert body["dry_run"] is True

    # stripe must NOT have been called
    assert len(charged_calls) == 0, "charge_off_session called during dry_run"

    # sub must NOT be advanced
    s = subs.get(cx, sid)
    assert s["order_count"] == 0, "subscription was advanced during dry_run"

    cx.close()


def test_failed_charge_bumps_failed_count_and_does_not_advance(monkeypatch):
    """A failed Stripe charge must bump failed_count and NOT advance the sub."""
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    _secret = _cron_secret()
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _secret)

    monkeypatch.setattr(
        appmod.stripe_pay, "charge_off_session",
        lambda *a, **k: {"id": None, "status": "failed", "decline_code": "insufficient_funds"},
    )
    _mock_price_cart(monkeypatch)
    _mock_qb(monkeypatch)
    _mock_ingest(monkeypatch)
    _mock_email(monkeypatch)

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    sid = _seed_sub(cx, next_charge_date="2000-01-01")

    c = appmod.app.test_client()
    r = c.post("/api/cron/charge-subscriptions", headers=_headers())
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["ok"] is True
    assert body["failed"] >= 1

    s = subs.get(cx, sid)
    # order_count must be unchanged
    assert s["order_count"] == 0, "subscription was advanced after failed charge"
    # failed_count must be incremented
    assert s.get("failed_count", 0) >= 1, "failed_count was not bumped"

    cx.close()
