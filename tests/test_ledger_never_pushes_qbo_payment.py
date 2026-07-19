"""The ledger must never CREATE a QBO Payment.

Every payment reaches QuickBooks on its own as a bank deposit (cards via
eProcessing/PayPal/Authorize.net, Zelle via the BofA feed), so a payment pushed from
the ledger duplicates money QBO is already receiving. On 2026-07-19 that had produced
6 duplicate QBO Payments that had to be deleted by hand.

These tests fail against the old behaviour, where add_payment() called
qbo_billing.record_payment by default.
"""
import sqlite3

import pytest

from dashboard import order_payments as OP
from dashboard import orders as O


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    OP.ensure_table(cx)
    return cx


def _order(cx, external_ref="INV-1"):
    return O.upsert_order(cx, source="funnel", external_ref=external_ref,
                          email="a@x.com", name="A", items=[], total_cents=10000)


@pytest.fixture
def no_qbo(monkeypatch):
    """Explode if anything tries to create a QBO payment."""
    calls = []

    def _boom(*a, **kw):
        calls.append((a, kw))
        raise AssertionError("qbo_billing.record_payment must never be called from the ledger")

    monkeypatch.setattr(OP.qbo_billing, "record_payment", _boom)
    return calls


def test_add_payment_does_not_create_a_qbo_payment(no_qbo):
    cx = _cx()
    oid = _order(cx)
    row = OP.add_payment(cx, oid, 5000, "Credit card (Stripe)", source="stripe")
    assert no_qbo == []                     # never reached QBO
    assert row["qbo_txn_id"] is None        # nothing to link — QBO gets it via the bank feed
    assert row["qbo_sync"] == "synced"      # terminal, so it cannot re-push later
    assert OP.balance(cx, oid)["paid_cents"] == 5000  # ledger still records the money


def test_linking_an_existing_qbo_txn_still_works(no_qbo):
    cx = _cx()
    oid = _order(cx)
    row = OP.add_payment(cx, oid, 5000, "Zelle", qbo_txn_id="24770")
    assert no_qbo == []
    assert row["qbo_txn_id"] == "24770"     # mirrors the real QBO txn
    assert row["qbo_sync"] == "synced"


def test_skip_qbo_push_is_vestigial_but_harmless(no_qbo):
    # Callers still pass it; it must not error and must behave like the default.
    cx = _cx()
    oid = _order(cx)
    row = OP.add_payment(cx, oid, 9239, "Zelle", skip_qbo_push=True)
    assert no_qbo == []
    assert row["qbo_txn_id"] is None
    assert row["qbo_sync"] == "synced"


def test_resync_on_a_payment_does_not_push_either(no_qbo):
    # resync() is the manual "try again" button; it must not become a back door.
    cx = _cx()
    oid = _order(cx)
    row = OP.add_payment(cx, oid, 5000, "Zelle")
    cx.execute("UPDATE order_payments SET qbo_sync='error', qbo_txn_id=NULL WHERE id=?",
               (row["id"],))
    cx.commit()
    OP.resync(cx, row["id"])
    assert no_qbo == []
    assert OP._row(cx, row["id"])["qbo_sync"] == "synced"


def test_no_qbo_invoice_is_no_longer_an_error_state(no_qbo):
    # Previously an order with no QBO invoice raised inside _push_payment and marked
    # the row 'error'. With invoicing retired that is the normal case, not a fault.
    cx = _cx()
    oid = _order(cx)
    # upsert_order requires an external_ref, so clear it afterwards to represent an
    # order with no QBO invoice behind it.
    cx.execute("UPDATE orders SET external_ref='' WHERE id=?", (oid,))
    cx.commit()
    row = OP.add_payment(cx, oid, 2500, "Check")
    assert no_qbo == []
    assert row["qbo_sync"] == "synced"
