"""Legacy pre-ledger paid orders must surface their payment + true balance.

A pre-ledger order can carry its payment on the ORDER row (orders.paid_cents,
pay_status='paid') with NO order_payments rows — e.g. #49 Bobbi Courtney, marked
Paid/Zelle before a ledger row existed. balance() must show that paid amount and
the real balance-due instead of "balance = full invoice total", so Edit Invoice,
the customer invoice and the board all agree. A single ledger row turns the
fallback off (the ledger wins), and the legacy amount must never be treated as
refundable (there is no real payment row to refund against).
"""
import sqlite3
import pytest
from dashboard import order_payments as op
from dashboard import orders


@pytest.fixture
def cx():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    orders.init_orders_table(c)
    op.ensure_table(c)
    # a $1181.97 order (mirrors #49's total)
    orders.upsert_order(c, source="inhouse", external_ref="INH-49",
                        email="aurainfusions@example.com", total_cents=118197)
    return c


def _oid(cx):
    return cx.execute("SELECT id FROM orders").fetchone()[0]


def _mark_legacy_paid(cx, oid, *, amount_cents, method="Zelle", status="paid"):
    # Set the legacy pay fields directly (as mark_order_paid_keep_status would),
    # without its portal-receipt side effects.
    cx.execute("UPDATE orders SET pay_status=?, pay_method=?, paid_cents=? WHERE id=?",
               (status, method, amount_cents, oid))
    cx.commit()


def test_legacy_paid_partial_surfaces_paid_and_balance(cx):
    oid = _oid(cx)
    _mark_legacy_paid(cx, oid, amount_cents=94947)  # Bobbi paid $949.47 via Zelle
    b = op.balance(cx, oid)
    assert b["paid_cents"] == 94947
    assert b["balance_cents"] == 23250          # $232.50 still due — not $1181.97
    assert b["legacy_fallback"] is True
    assert b["ledger_paid_cents"] == 0          # no real ledger payment


def test_legacy_paid_in_full_reads_zero_balance(cx):
    oid = _oid(cx)
    _mark_legacy_paid(cx, oid, amount_cents=118197)  # paid in full (mirrors #63)
    b = op.balance(cx, oid)
    assert b["paid_cents"] == 118197
    assert b["balance_cents"] == 0
    assert b["legacy_fallback"] is True


def test_ledger_row_wins_over_legacy_field(cx):
    oid = _oid(cx)
    _mark_legacy_paid(cx, oid, amount_cents=94947)
    op.add_payment(cx, oid, 25000, "Zelle")   # a real ledger payment appears
    b = op.balance(cx, oid)
    assert b["paid_cents"] == 25000            # ledger wins, legacy ignored
    assert b["balance_cents"] == 93197
    assert b["legacy_fallback"] is False
    assert b["ledger_paid_cents"] == 25000


def test_no_fallback_when_not_marked_paid(cx):
    oid = _oid(cx)
    # claimed (customer says they sent, owner not confirmed) → no paid_cents/no fallback
    _mark_legacy_paid(cx, oid, amount_cents=0, method="Zelle", status="claimed")
    b = op.balance(cx, oid)
    assert b["paid_cents"] == 0
    assert b["balance_cents"] == 118197
    assert b["legacy_fallback"] is False


def test_legacy_paid_is_not_refundable_via_ledger(cx):
    oid = _oid(cx)
    _mark_legacy_paid(cx, oid, amount_cents=94947)
    # No real ledger payment exists, so nothing is refundable through the ledger
    # (refunding a legacy order is done after it is backfilled into the ledger).
    assert op.refundable_cents(cx, oid) == 0


def test_backfilled_row_is_refundable(cx):
    oid = _oid(cx)
    op.add_payment(cx, oid, 94947, "Zelle", source="legacy")
    assert op.refundable_cents(cx, oid) == 94947
