"""Trial -> full conversion grant (PR2): the first $99 membership charge hands the
accrued volume discount back as points.

Mirrors test_membership_charge_cron.py: seeds a due kind='membership' subscription
(order_count=0 = still on trial) plus the buyer's in-window remedy orders against
the real LOG_DB, stubs externals, posts to the charge cron, and asserts a single
`trial_upgrade_credit` points entry equal to the accrued discount is written.
"""
import json
import os
import sqlite3

import app as appmod
from dashboard import subscriptions as subs
from dashboard import orders as orders_mod
from dashboard import points as points_mod
from dashboard import trial_credit as tc

EMAIL = "trial-grant@example.com"
# Bone Builder is qty-eligible ($69.97, qty_pricing). At qty 3 the member tier is
# 5997 vs the 6997 regular -> (6997-5997)*3 = 3000 cents accrued.
EXPECTED_CREDIT = 3000


def _cron_secret():
    return os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET", "test-secret")


def _headers():
    return {"X-Cron-Secret": _cron_secret()}


def _enable(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    if not (os.environ.get("CRON_SECRET") or os.environ.get("CONSOLE_SECRET")):
        monkeypatch.setenv("CONSOLE_SECRET", _cron_secret())


def _stub_externals(monkeypatch):
    monkeypatch.setattr(appmod.stripe_pay, "charge_off_session",
                        lambda *a, **k: {"status": "succeeded", "id": "pi_x"})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1", "DisplayName": "x"})
    monkeypatch.setattr(appmod.qb, "create_invoice", lambda *a, **k: {"Id": "INV1"})
    monkeypatch.setattr(appmod, "_send_subscription_email", lambda *a, **k: ("smtp", None))
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)


def _clean(cx, email):
    cx.execute("DELETE FROM subscriptions WHERE email=?", (email,))
    cx.execute("DELETE FROM orders WHERE lower(email)=?", (email.lower(),))
    cx.execute("DELETE FROM points_ledger WHERE email=?", (email,))
    cx.commit()


def _seed_trial_with_orders(cx, email=EMAIL, *, item_qty=3):
    """A trial buyer: a due membership sub (order_count=0) + an in-window
    biofield_trial order + a reorder with a qty-eligible line."""
    subs.init_subscriptions_table(cx)
    subs.migrate_add_failed_count(cx)
    subs.migrate_add_membership_columns(cx)
    orders_mod.init_orders_table(cx)
    points_mod.init_points_table(cx)
    _clean(cx, email)
    from datetime import date
    today = date.today().isoformat()
    # Trial purchase = window start (today, so the reorder is in window).
    cx.execute("INSERT INTO orders (created_at, source, external_ref, email, items_json, "
               "total_cents, status) VALUES (?, 'biofield_trial', ?, ?, ?, 100, 'new')",
               (today + "T00:00:00+00:00", f"bt-{email}", email,
                json.dumps([{"name": "Biofield Analysis", "qty": 1}])))
    cx.execute("INSERT INTO orders (created_at, source, external_ref, email, items_json, "
               "total_cents, status) VALUES (?, 'reorder', ?, ?, ?, 0, 'new')",
               (today + "T01:00:00+00:00", f"re-{email}", email,
                json.dumps([{"name": "Bone Builder", "qty": item_qty}])))
    sid = subs.create_membership(
        cx, email=email, stripe_customer_id="cus_tc", stripe_payment_method_id="pm_tc",
        amount_cents=9900, next_charge_date=today, cadence_months=1)
    cx.commit()
    return sid


def test_first_charge_grants_accrued_credit_as_points(monkeypatch):
    _enable(monkeypatch)
    _stub_externals(monkeypatch)
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    sid = _seed_trial_with_orders(cx)

    # Sanity: order_count starts at 0 (trial) and accrual computes the expected amount.
    assert subs.category_for(cx, EMAIL) == "trial"
    assert appmod._trial_credit_for_email(cx, EMAIL) == EXPECTED_CREDIT

    r = appmod.app.test_client().post("/api/cron/charge-subscriptions", headers=_headers())
    assert r.status_code == 200, r.data
    assert r.get_json()["charged"] >= 1

    # Exactly one trial_upgrade_credit ledger row of the accrued amount.
    rows = cx.execute(
        "SELECT delta_cents FROM points_ledger WHERE email=? AND reason=?",
        (EMAIL, tc.CREDIT_REASON)).fetchall()
    assert len(rows) == 1
    assert rows[0]["delta_cents"] == EXPECTED_CREDIT
    assert points_mod.balance(cx, EMAIL) == EXPECTED_CREDIT
    # The buyer is now full (order_count bumped 0 -> 1).
    assert subs.category_for(cx, EMAIL) == "full"

    _clean(cx, EMAIL); cx.close()


def test_second_run_is_a_no_op(monkeypatch):
    _enable(monkeypatch)
    _stub_externals(monkeypatch)
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    _seed_trial_with_orders(cx)
    client = appmod.app.test_client()

    client.post("/api/cron/charge-subscriptions", headers=_headers())
    # Re-run: the sub is no longer due (date advanced) AND no longer a trial
    # (order_count==1), so no second credit is written.
    client.post("/api/cron/charge-subscriptions", headers=_headers())

    rows = cx.execute(
        "SELECT 1 FROM points_ledger WHERE email=? AND reason=?",
        (EMAIL, tc.CREDIT_REASON)).fetchall()
    assert len(rows) == 1, "conversion credit must be granted exactly once"

    _clean(cx, EMAIL); cx.close()


def test_credit_path_is_idempotent_on_repeat_grant(monkeypatch):
    """The load-bearing safety property: even if the credit path runs twice for the
    same buyer (e.g. a second kind=membership sub hitting order_count==0->1 on its
    own first charge), the deterministic per-email order_ref + has_entry dedup means
    exactly one ledger row is ever written."""
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    _seed_trial_with_orders(cx)
    points_mod.init_points_table(cx)

    ref = tc.credit_order_ref(EMAIL)
    points_mod.credit(cx, EMAIL, value_cents=EXPECTED_CREDIT, reason=tc.CREDIT_REASON, order_ref=ref)
    points_mod.credit(cx, EMAIL, value_cents=EXPECTED_CREDIT, reason=tc.CREDIT_REASON, order_ref=ref)

    rows = cx.execute("SELECT delta_cents FROM points_ledger WHERE email=? AND reason=?",
                      (EMAIL, tc.CREDIT_REASON)).fetchall()
    assert len(rows) == 1
    assert rows[0]["delta_cents"] == EXPECTED_CREDIT

    _clean(cx, EMAIL); cx.close()


def test_cancelled_before_charge_gets_nothing(monkeypatch):
    _enable(monkeypatch)
    _stub_externals(monkeypatch)
    cx = sqlite3.connect(appmod.LOG_DB); cx.row_factory = sqlite3.Row
    sid = _seed_trial_with_orders(cx)
    subs.set_status(cx, sid, "cancelled")   # trial cancelled before the first $99 clears

    appmod.app.test_client().post("/api/cron/charge-subscriptions", headers=_headers())

    rows = cx.execute(
        "SELECT 1 FROM points_ledger WHERE email=? AND reason=?",
        (EMAIL, tc.CREDIT_REASON)).fetchall()
    assert rows == [], "a cancelled-before-charge trial buyer accrues no credit"

    _clean(cx, EMAIL); cx.close()
