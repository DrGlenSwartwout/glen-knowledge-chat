"""Stage 2.5: finance.refund_order works for paid-only orders (no QBO invoice)."""
import pytest
from dashboard import finance, qbo_billing, orders as O
from dashboard import stripe_pay


def _patch_common(monkeypatch, *, invoice, order, refund_calls, cust_calls, stripe_calls,
                   order_calls=None, call_order=None):
    monkeypatch.setattr(qbo_billing, "get_invoice", lambda iid: invoice)

    def fake_cust(email, name=""):
        cust_calls.append((email, name)); return {"Id": "CUST9"}
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", fake_cust)

    def fake_refund(customer_id, amount, *, description="Refund", **kw):
        refund_calls.append({"customer_id": customer_id, "amount": amount,
                             "description": description})
        if call_order is not None:
            call_order.append("refund_receipt")
        return {"Id": "RR1", "DocNumber": "R-100"}
    monkeypatch.setattr(qbo_billing, "create_refund_receipt", fake_refund)

    def fake_stripe(pi, cents):
        stripe_calls.append((pi, cents))
        if call_order is not None:
            call_order.append("stripe_refund")
        return {"id": "re_1"}
    monkeypatch.setattr(stripe_pay, "refund", fake_stripe)

    def fake_get_order(cx, oid):
        if order_calls is not None:
            order_calls.append(oid)
        return order
    monkeypatch.setattr(O, "get_order", fake_get_order)
    monkeypatch.setattr(O, "find_order_by_external_ref", lambda cx, ref: order)


def test_paid_only_refund_books_refundreceipt_without_invoice(monkeypatch):
    order = {"id": 7, "external_ref": "tok-abc", "email": "a@b.com", "name": "A",
             "qbo_sales_receipt_id": "SR5", "stripe_payment_intent": ""}
    refund_calls, cust_calls, stripe_calls = [], [], []
    _patch_common(monkeypatch, invoice=None, order=order, refund_calls=refund_calls,
                  cust_calls=cust_calls, stripe_calls=stripe_calls)
    res = finance._refund_order_exec({"order_id": 7, "amount": 25.0}, {"cx": object()})
    assert cust_calls == [("a@b.com", "A")]          # resolved customer by email
    assert len(refund_calls) == 1
    assert refund_calls[0]["customer_id"] == "CUST9"
    assert refund_calls[0]["amount"] == 25.0
    assert "SR5" in refund_calls[0]["description"]    # traceability
    assert res["refund_receipt_id"] == "RR1"
    assert stripe_calls == []                         # no PI on file -> no card refund


def test_paid_only_refund_also_refunds_card_when_pi_present(monkeypatch):
    order = {"id": 8, "external_ref": "tok-xyz", "email": "c@b.com", "name": "C",
             "qbo_sales_receipt_id": "SR6", "stripe_payment_intent": "pi_123"}
    refund_calls, cust_calls, stripe_calls = [], [], []
    call_order = []
    _patch_common(monkeypatch, invoice=None, order=order, refund_calls=refund_calls,
                  cust_calls=cust_calls, stripe_calls=stripe_calls, call_order=call_order)
    res = finance._refund_order_exec({"order_id": 8, "amount": 30.0}, {"cx": object()})
    assert stripe_calls == [("pi_123", 3000)]         # card refunded first
    assert len(refund_calls) == 1
    assert res["stripe_refund"] is True
    # Directly assert ordering: the Stripe card refund must fire before the QBO
    # RefundReceipt is booked (only book QBO if the card refund actually succeeded).
    assert call_order == ["stripe_refund", "refund_receipt"]


def test_stuck_pending_order_still_refunds(monkeypatch):
    order = {"id": 9, "external_ref": "tok-p", "email": "d@b.com", "name": "D",
             "qbo_sales_receipt_id": "PENDING", "stripe_payment_intent": ""}
    refund_calls, cust_calls, stripe_calls = [], [], []
    _patch_common(monkeypatch, invoice=None, order=order, refund_calls=refund_calls,
                  cust_calls=cust_calls, stripe_calls=stripe_calls)
    res = finance._refund_order_exec({"order_id": 9, "amount": 10.0}, {"cx": object()})
    assert len(refund_calls) == 1                     # refunds despite PENDING
    assert res["refund_receipt_id"] == "RR1"


def test_legacy_invoice_path_unchanged(monkeypatch):
    # get_invoice returns a real invoice -> customer from CustomerRef, NOT find_or_create_customer.
    inv = {"CustomerRef": {"value": "LEGACY42"}, "DocNumber": "1001"}
    order = {"id": 10, "external_ref": "1001", "email": "e@b.com", "name": "E",
             "qbo_sales_receipt_id": None, "stripe_payment_intent": ""}
    refund_calls, cust_calls, stripe_calls = [], [], []
    _patch_common(monkeypatch, invoice=inv, order=order, refund_calls=refund_calls,
                  cust_calls=cust_calls, stripe_calls=stripe_calls)
    res = finance._refund_order_exec({"invoice_id": "1001", "amount": 5.0}, {"cx": object()})
    assert cust_calls == []                            # legacy path must NOT resolve by email
    assert refund_calls[0]["customer_id"] == "LEGACY42"


def test_unresolvable_still_raises(monkeypatch):
    # No invoice, and the order has no email -> cannot resolve a customer.
    order = {"id": 11, "external_ref": "tok-none", "email": "", "name": "",
             "qbo_sales_receipt_id": "SR7", "stripe_payment_intent": ""}
    refund_calls, cust_calls, stripe_calls = [], [], []
    _patch_common(monkeypatch, invoice=None, order=order, refund_calls=refund_calls,
                  cust_calls=cust_calls, stripe_calls=stripe_calls)
    with pytest.raises(ValueError):
        finance._refund_order_exec({"order_id": 11, "amount": 5.0}, {"cx": object()})


def test_valid_invoice_id_with_stale_order_id_still_succeeds(monkeypatch):
    """A caller passing BOTH a valid invoice_id AND a stale/invalid order_id (get_order
    returns None for it) must still succeed via the invoice_id path, not raise
    "order not found". This is the dormant case Fix 1 restores: the order-not-found
    raise must only fire when there is no invoice_id to fall back on."""
    inv = {"CustomerRef": {"value": "LEGACY99"}, "DocNumber": "2002"}
    refund_calls, cust_calls, stripe_calls = [], [], []
    _patch_common(monkeypatch, invoice=inv, order=None, refund_calls=refund_calls,
                  cust_calls=cust_calls, stripe_calls=stripe_calls)
    res = finance._refund_order_exec(
        {"invoice_id": "2002", "order_id": 999999, "amount": 12.5}, {"cx": object()})
    assert cust_calls == []                            # resolved via invoice, not email
    assert refund_calls[0]["customer_id"] == "LEGACY99"
    assert res["refund_receipt_id"] == "RR1"
