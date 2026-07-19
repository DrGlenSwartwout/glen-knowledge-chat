"""Recording a payment that ALREADY exists in QBO must not create a second one.

Regression cover for a live double-credit: backfilling Dana Tamraz's order #66
card payment into the ledger pushed a duplicate QBO payment (24772) alongside
the real one (24458). These tests pin that qbo_txn_id / skip_qbo_push suppress
the push, and that the DEFAULT still pushes (so normal payments keep working).

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


def test_default_still_pushes_to_qbo(cx, pushes):
    # The normal path must be unchanged — a genuinely new payment books in QBO.
    row = op.add_payment(cx, 66, 9239, "Zelle")
    assert len(pushes) == 1
    assert pushes[0]["amount_cents"] == 9239
    assert row["qbo_txn_id"] == "NEW-QBO-ID"
    assert row["qbo_sync"] == "synced"


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
