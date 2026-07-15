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


def test_add_payment_syncs_to_qbo(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "412.82"})
    calls = {}
    monkeypatch.setattr(qbo_billing, "record_payment",
                        lambda cid, amt, iid, method=None: calls.update(
                            cid=cid, amt=amt, iid=iid, method=method) or {"Id": "P9"})
    oid = _oid(cx)
    row = op.add_payment(cx, oid, 13100, "Zelle")
    assert row["qbo_txn_id"] == "P9"
    assert row["qbo_sync"] == "synced"
    assert calls == {"cid": "42", "amt": 13100, "iid": "INV-1", "method": "Zelle"}


def test_void_excludes_from_balance_and_calls_qbo(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice",
                        lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "1"})
    monkeypatch.setattr(qbo_billing, "record_payment",
                        lambda *a, **k: {"Id": "P9"})
    voided = {}
    monkeypatch.setattr(qbo_billing, "void_payment",
                        lambda txn: voided.update(txn=txn))
    oid = _oid(cx)
    row = op.add_payment(cx, oid, 13100, "Zelle")
    op.void(cx, row["id"], "keyed wrong amount")
    assert op.balance(cx, oid)["paid_cents"] == 0
    assert voided == {"txn": "P9"}
    # voiding again is a no-op (no second QBO call)
    voided.clear()
    op.void(cx, row["id"], "again")
    assert voided == {}


def test_void_null_txn_skips_qbo(cx, monkeypatch):
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice", lambda iid: None)  # push fails
    called = {"n": 0}
    monkeypatch.setattr(qbo_billing, "void_payment",
                        lambda txn: called.__setitem__("n", called["n"] + 1))
    oid = _oid(cx)
    row = op.add_payment(cx, oid, 100, "Cash")   # qbo_sync becomes 'error', no txn
    assert row["qbo_sync"] == "error" and row["qbo_txn_id"] is None
    op.void(cx, row["id"], "typo")
    assert called["n"] == 0
