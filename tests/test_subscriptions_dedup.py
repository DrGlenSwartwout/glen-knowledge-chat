import sqlite3
from unittest.mock import patch
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


def test_create_once_atomic_backstop_when_precheck_lies():
    """Simulates the real race: redirect + webhook processes both pass the
    has_subscription_for_order pre-check (e.g. interleaved before either commits),
    so the dedup must be enforced by the UNIQUE index + IntegrityError catch,
    not just the SELECT-then-INSERT pre-check."""
    cx = _mk()
    first = subs.create_once(cx, order_ref="tokRace", **_KW)
    assert first is not None

    # Bypass the pre-check entirely to prove the atomic backstop (UNIQUE index +
    # IntegrityError catch) is what stops the duplicate, not the SELECT guard.
    with patch.object(subs, "has_subscription_for_order", return_value=False):
        second = subs.create_once(cx, order_ref="tokRace", **_KW)

    assert second is None
    n = cx.execute(
        "SELECT COUNT(*) FROM subscriptions WHERE order_ref='tokRace'"
    ).fetchone()[0]
    assert n == 1


def test_unique_index_allows_multiple_null_order_ref_rows():
    """NULL order_ref rows (pre-migration / non-order-linked subs) must not
    collide under the UNIQUE index — SQLite treats each NULL as distinct."""
    cx = _mk()
    kw = dict(_KW)
    id1 = subs.create(cx, **kw)  # order_ref defaults to None
    id2 = subs.create(cx, **kw)  # also None — must NOT raise IntegrityError
    assert id1 != id2
    n = cx.execute(
        "SELECT COUNT(*) FROM subscriptions WHERE order_ref IS NULL"
    ).fetchone()[0]
    assert n == 2
