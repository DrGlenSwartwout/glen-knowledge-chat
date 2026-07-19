"""The ledger must never create a QBO payment.

Originally regression cover for a live double-credit: backfilling Dana Tamraz's
order #66 card payment pushed a duplicate QBO payment (24772) alongside the real
one (24458). At the time only qbo_txn_id / skip_qbo_push suppressed the push and
the DEFAULT still pushed.

That default is now gone. Every payment reaches QuickBooks on its own as a BANK
DEPOSIT (cards via eProcessing/PayPal/Authorize.net, Zelle via the BofA feed), so
any push is a duplicate -- the "normal payment" case was itself the bug. Six such
duplicates were deleted by hand on 2026-07-19. Note the scope: QBO is retired for
INVOICING, not as a system; it still gets the money, from the bank feed.

Hermetic: in-memory sqlite + monkeypatched qbo_billing. No secrets, no network.
"""
import sqlite3

import pytest

from dashboard import order_payments as op


@pytest.fixture()
def cx():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    op.ensure_table(c)
    return c


@pytest.fixture()
def pushes(monkeypatch):
    """Capture QBO payment-creation calls."""
    calls = []

    def fake_record_payment(cid, amount_cents, inv_id, method=None):
        calls.append({"cid": cid, "amount_cents": amount_cents,
                      "inv_id": inv_id, "method": method})
        return {"Id": "NEW-QBO-ID"}

    monkeypatch.setattr(op.qbo_billing, "record_payment", fake_record_payment)
    # A resolvable QBO context, so a push WOULD succeed if attempted.
    monkeypatch.setattr(op, "_qbo_ctx", lambda cx, oid: ("cust-1", "inv-1"))
    return calls


def test_linking_existing_qbo_txn_does_not_push(cx, pushes):
    row = op.add_payment(cx, 66, 22291, "Credit card (Stripe)",
                         qbo_txn_id="24458")
    assert pushes == []                       # nothing created in QBO
    assert row["qbo_txn_id"] == "24458"        # linked to the REAL txn
    assert row["qbo_sync"] == "synced"


def test_skip_flag_marks_synced_with_null_txn_and_does_not_push(cx, pushes):
    row = op.add_payment(cx, 66, 13100, "Zelle", skip_qbo_push=True)
    assert pushes == []
    assert row["qbo_txn_id"] is None
    assert row["qbo_sync"] == "synced"


def test_default_does_not_push_to_qbo(cx, pushes):
    # Inverted deliberately. This used to assert the default PUSHED; that default was
    # the duplicate factory. Note the fixture gives a resolvable QBO context, so a push
    # WOULD succeed if one were attempted — nothing here is masking it.
    row = op.add_payment(cx, 66, 9239, "Zelle")
    assert pushes == []                 # QBO gets this money via the bank deposit
    assert row["qbo_txn_id"] is None    # nothing to mirror
    assert row["qbo_sync"] == "synced"  # terminal, so it cannot re-push later


def test_balance_counts_nonpushed_payments(cx, pushes, monkeypatch):
    # balance() reads the order for its invoice total; stub it (order #66's real
    # invoice is $534.30) so this stays a pure ledger test.
    monkeypatch.setattr(op.orders, "get_order", lambda cx, oid: {"total_cents": 53430})
    op.add_payment(cx, 66, 22291, "Credit card (Stripe)", qbo_txn_id="24458")
    op.add_payment(cx, 66, 13100, "Zelle", qbo_txn_id="24770")
    op.add_payment(cx, 66, 8800, "Zelle", qbo_txn_id="24771")
    assert pushes == []
    b = op.balance(cx, 66)
    assert b["paid_cents"] == 44191
    assert b["balance_cents"] == 53430 - 44191   # the $92.39 still outstanding


def test_amount_must_be_positive(cx, pushes):
    with pytest.raises(ValueError):
        op.add_payment(cx, 66, 0, "Zelle", skip_qbo_push=True)
