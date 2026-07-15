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
    # a $412.82 order
    orders.upsert_order(c, source="qbo", external_ref="INV-1",
                        email="dana@example.com", total_cents=41282)
    return c


def _oid(cx):
    return cx.execute("SELECT id FROM orders").fetchone()[0]


def test_add_payment_reduces_balance(cx):
    oid = _oid(cx)
    op.add_payment(cx, oid, 22291, "Credit card (Stripe)", source="stripe",
                   external_ref="pi_1")
    op.add_payment(cx, oid, 13100, "Zelle")
    b = op.balance(cx, oid)
    assert b["paid_cents"] == 35391
    assert b["refunded_cents"] == 0
    assert b["invoice_cents"] == 41282
    assert b["balance_cents"] == 5891


def test_overpayment_is_negative_balance(cx):
    oid = _oid(cx)
    op.add_payment(cx, oid, 50000, "Zelle")
    assert op.balance(cx, oid)["balance_cents"] == -8718


def test_stripe_payment_idempotent_on_external_ref(cx):
    oid = _oid(cx)
    op.add_payment(cx, oid, 22291, "Credit card (Stripe)", source="stripe",
                   external_ref="pi_1")
    op.add_payment(cx, oid, 22291, "Credit card (Stripe)", source="stripe",
                   external_ref="pi_1")
    rows = op.list_payments(cx, oid)
    assert len([r for r in rows if r["kind"] == "payment"]) == 1
