"""Tests for /begin/checkout-return — group-bundle membership grant (Task 4).

When a paid Stripe session carries grant_group_months>0 in its metadata AND
GROUP_BUNDLE_ENABLED is set, the checkout-return handler vaults the buyer's
card (from the PaymentIntent) and creates a kind='membership' subscription
whose first charge lands after the free window (today + N months).

Scenarios:
  1. flag on, first grant     → membership created, $99/mo, cadence 1, charge=+N months.
  2. flag on, literal re-run   → idempotent (still exactly 1 membership).
  3. flag on, existing member  → window pushed out by N months (no 2nd row).
  4. flag unset                → NO membership created.
"""

import sqlite3
from datetime import date

import app as appmod
from dashboard import stripe_pay as _stripe_pay_mod
from dashboard import subscriptions as subs


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_session(grant_months="3", email="p@x.com", invoice_id="INVG1",
                  payment_status="paid"):
    return {
        "payment_status": payment_status,
        "payment_intent": "pi_test",
        "amount_total": 7000,
        "metadata": {
            "grant_group_months": str(grant_months),
            "email": email,
            "invoice_id": invoice_id,
        },
    }


def _wire(monkeypatch, tmp_path, session):
    """Point LOG_DB at a temp db, stub get_session + get_payment_intent, and
    silence every unrelated side effect in the return handler."""
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)

    monkeypatch.setattr(_stripe_pay_mod, "get_session", lambda sid: session)
    monkeypatch.setattr(
        _stripe_pay_mod, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1"})

    # Silence non-grant side effects (no kind=subscribe/client/retail metadata here,
    # but the points/referral settlers + bos lookups guard on _o_for_points which is
    # None when find_order_by_external_ref returns None — keep them inert anyway).
    monkeypatch.setattr(appmod._bos_orders, "find_order_by_external_ref",
                        lambda *a, **k: None)
    monkeypatch.setattr(appmod._bos_orders, "set_order_stripe_pi",
                        lambda *a, **k: None)
    return db


def _active(db, email="p@x.com"):
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        subs.init_subscriptions_table(cx)
        subs.migrate_add_membership_columns(cx)
        return subs.active_memberships_by_email(cx, email)


def _get(monkeypatch):
    appmod.app.config["TESTING"] = True
    client = appmod.app.test_client()
    return client.get("/begin/checkout-return?session_id=cs_test")


# ── Test 1: flag on, first grant → membership created ─────────────────────────

def test_grant_creates_membership(monkeypatch, tmp_path):
    monkeypatch.setenv("GROUP_BUNDLE_ENABLED", "1")
    db = _wire(monkeypatch, tmp_path, _make_session(grant_months="3"))

    resp = _get(monkeypatch)
    assert resp.status_code in (301, 302, 303, 307, 308)

    rows = _active(db)
    assert len(rows) == 1
    m = rows[0]
    assert m["kind"] == "membership"
    assert m["amount_cents"] == 9900
    assert m["cadence_months"] == 1
    expected = subs.add_months(date.today().isoformat(), 3)
    assert m["next_charge_date"] == expected


# ── Test 2: literal re-run of the SAME return is idempotent ───────────────────

def test_grant_rerun_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("GROUP_BUNDLE_ENABLED", "1")
    db = _wire(monkeypatch, tmp_path, _make_session(grant_months="3", invoice_id="INVG1"))

    _get(monkeypatch)
    first = _active(db)
    assert len(first) == 1
    first_charge = first[0]["next_charge_date"]

    # Same return again — must NOT create a second row NOR push the date out.
    _get(monkeypatch)
    again = _active(db)
    assert len(again) == 1
    assert again[0]["next_charge_date"] == first_charge


# ── Test 3: a genuinely new order extends an existing membership ──────────────

def test_new_grant_extends_existing_window(monkeypatch, tmp_path):
    monkeypatch.setenv("GROUP_BUNDLE_ENABLED", "1")
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    monkeypatch.setattr(appmod._bos_orders, "find_order_by_external_ref",
                        lambda *a, **k: None)
    monkeypatch.setattr(appmod._bos_orders, "set_order_stripe_pi",
                        lambda *a, **k: None)
    monkeypatch.setattr(
        _stripe_pay_mod, "get_payment_intent",
        lambda pi: {"customer": "cus_1", "payment_method": "pm_1"})

    # Seed an existing active membership directly.
    start = date.today().isoformat()
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        subs.init_subscriptions_table(cx)
        subs.migrate_add_membership_columns(cx)
        subs.create_membership(
            cx, email="p@x.com", stripe_customer_id="cus_old",
            stripe_payment_method_id="pm_old", amount_cents=9900,
            next_charge_date=start)
    prior = _active(db)[0]["next_charge_date"]

    # New program order (DIFFERENT invoice) grants 2 more months → window extends.
    monkeypatch.setattr(_stripe_pay_mod, "get_session",
                        lambda sid: _make_session(grant_months="2", invoice_id="INVG_NEW"))
    _get(monkeypatch)

    rows = _active(db)
    assert len(rows) == 1, "must extend, not create a second membership"
    assert rows[0]["next_charge_date"] == subs.add_months(prior, 2)


# ── Test 4: flag unset → NO membership ────────────────────────────────────────

def test_no_grant_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.delenv("GROUP_BUNDLE_ENABLED", raising=False)
    db = _wire(monkeypatch, tmp_path, _make_session(grant_months="3"))

    resp = _get(monkeypatch)
    assert resp.status_code in (301, 302, 303, 307, 308)
    assert _active(db) == []
