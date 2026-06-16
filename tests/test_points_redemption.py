"""In-house order points settlement: orders.settle_order_points redeems applied
points + earns on full-price spend, idempotently per external_ref, against the
real ledger. This is what wires the Phase 1 'points off the total' to an actual
balance decrement on payment (Zelle/Wise via record_payment, card via checkout-return)."""
import sqlite3

from dashboard import orders as O
from dashboard import points


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    points.init_points_table(cx)
    return cx


def _order(cx, *, email="a@x.com", points_redeemed_cents=0, discount_cents=0,
           total_cents=5000, shipping_cents=0, get_cents=0, ref="INH-T1"):
    oid = O.upsert_order(cx, source="in-house", external_ref=ref, status="proposed",
                         email=email, total_cents=total_cents, discount_cents=discount_cents,
                         shipping_cents=shipping_cents, get_cents=get_cents,
                         points_redeemed_cents=points_redeemed_cents)
    return O.get_order(cx, oid)


def test_settle_redeems_applied_points_once():
    cx = _cx()
    points.earn(cx, "a@x.com", full_price_cents=20000, earn_pct=0.05, order_ref="seed")  # 1000
    order = _order(cx, points_redeemed_cents=200, total_cents=4800)
    O.settle_order_points(cx, order)
    assert points.balance(cx, "a@x.com") == 800        # 1000 - 200 redeemed
    # idempotent on external_ref — a second settle (e.g. both pay paths) does not double-decrement
    O.settle_order_points(cx, order)
    assert points.balance(cx, "a@x.com") == 800


def test_settle_earns_on_full_price_order():
    cx = _cx()
    order = _order(cx, total_cents=7265, shipping_cents=1265, ref="INH-E1")
    O.settle_order_points(cx, order)
    assert points.balance(cx, "a@x.com") == 300        # product 6000 * 5%
    O.settle_order_points(cx, order)
    assert points.balance(cx, "a@x.com") == 300        # idempotent


def test_settle_no_earn_when_points_used_or_discounted():
    cx = _cx()
    points.earn(cx, "a@x.com", full_price_cents=20000, earn_pct=0.05, order_ref="seed")  # 1000
    # points used → not a full-price order → redeem only, no earn
    order = _order(cx, points_redeemed_cents=300, total_cents=4700, ref="INH-R1")
    O.settle_order_points(cx, order)
    assert points.balance(cx, "a@x.com") == 700


def test_settle_redeem_over_balance_does_not_crash_or_decrement():
    cx = _cx()
    points.earn(cx, "a@x.com", full_price_cents=2000, earn_pct=0.05, order_ref="seed")  # 100
    order = _order(cx, points_redeemed_cents=500, total_cents=4500, ref="INH-O1")
    O.settle_order_points(cx, order)                   # redeem 500 > balance 100 → swallowed
    assert points.balance(cx, "a@x.com") == 100        # unchanged, no exception


def test_settle_noop_without_email():
    cx = _cx()
    order = _order(cx, email="", points_redeemed_cents=200, ref="INH-N1")
    O.settle_order_points(cx, order)                   # must not raise
    assert points.balance(cx, "") == 0
