import sqlite3
from dashboard import subscriptions as subs

def _mk():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    subs.init_subscriptions_table(cx)
    return cx

_KW = dict(email="a@b.com", stripe_customer_id="cus_1", stripe_payment_method_id="pm_1",
           items=[{"slug": "x", "qty": 1}], cadence_months=1, ship_address={}, next_charge_date="2026-08-16")

def test_create_once_dedups_on_order_ref():
    cx = _mk()
    first = subs.create_once(cx, order_ref="tok1", **_KW)
    second = subs.create_once(cx, order_ref="tok1", **_KW)
    assert first is not None
    assert second is None
    n = cx.execute("SELECT COUNT(*) FROM subscriptions WHERE order_ref='tok1'").fetchone()[0]
    assert n == 1

def test_has_subscription_for_order():
    cx = _mk()
    assert subs.has_subscription_for_order(cx, "tok9") is False
    subs.create_once(cx, order_ref="tok9", **_KW)
    assert subs.has_subscription_for_order(cx, "tok9") is True

def test_create_once_distinct_refs_both_insert():
    cx = _mk()
    assert subs.create_once(cx, order_ref="tokA", **_KW) is not None
    assert subs.create_once(cx, order_ref="tokB", **_KW) is not None
    assert cx.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 2
