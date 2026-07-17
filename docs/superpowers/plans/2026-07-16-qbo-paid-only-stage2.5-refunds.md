# QBO Paid-Only Stage 2.5 (Refunds) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the console refund a paid-only order (no QBO invoice) by resolving the customer via `find_or_create_customer` and recording a QBO RefundReceipt, instead of raising "invoice not found".

**Architecture:** One branch in `dashboard/finance.py:_refund_order_exec` — when `get_invoice` returns None but the order is resolvable (paid-only), resolve `customer_id` by email; the existing Stripe-refund + `create_refund_receipt` flow is unchanged.

**Tech Stack:** Python, pytest, QuickBooks Online REST API, Stripe.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-16-qbo-paid-only-stage2.5-refunds-design.md`.
- Do NOT change the legacy invoice path (orders that DO resolve via `get_invoice` must behave exactly as today) — only ADD a fallback.
- Do NOT change the Stripe card-refund mechanics, `create_refund_receipt`, or partial-refund behavior.
- Money-path: card refund happens FIRST; the QBO RefundReceipt is recorded only after (unchanged ordering).
- Customer resolution for paid-only orders is by EMAIL via `find_or_create_customer` (idempotent — same customer that booked the SalesReceipt). Works even when `qbo_sales_receipt_id == 'PENDING'`.
- `void_invoice` is out of scope — do not touch `_void_invoice_exec`.
- Run tests via `doppler run --config dev -- python3 -m pytest <file>` (default doppler config is prd/DATA_DIR=/data and breaks app-import collection). This test file imports only `dashboard.finance`/`qbo_billing`/`orders`, so it may not need the app, but use `--config dev` for consistency.

---

## Task 1: Paid-only fallback in `_refund_order_exec`

**Files:**
- Modify: `dashboard/finance.py` — `_refund_order_exec` (~272-320)
- Test: `tests/test_finance_refund_paid_only.py` (create)

**Interfaces:**
- Consumes: `qbo_billing.get_invoice`, `qbo_billing.find_or_create_customer`, `qbo_billing.create_refund_receipt`, `dashboard.orders.get_order`, `dashboard.orders.find_order_by_external_ref`, `dashboard.stripe_pay.refund`.
- Produces: no new signature — `_refund_order_exec(params, ctx)` unchanged externally; adds a customer-resolution fallback for paid-only orders.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_finance_refund_paid_only.py`:

```python
"""Stage 2.5: finance.refund_order works for paid-only orders (no QBO invoice)."""
import pytest
from dashboard import finance, qbo_billing, orders as O
from dashboard import stripe_pay


def _patch_common(monkeypatch, *, invoice, order, refund_calls, cust_calls, stripe_calls):
    monkeypatch.setattr(qbo_billing, "get_invoice", lambda iid: invoice)

    def fake_cust(email, name=""):
        cust_calls.append((email, name)); return {"Id": "CUST9"}
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", fake_cust)

    def fake_refund(customer_id, amount, *, description="Refund", **kw):
        refund_calls.append({"customer_id": customer_id, "amount": amount,
                             "description": description})
        return {"Id": "RR1", "DocNumber": "R-100"}
    monkeypatch.setattr(qbo_billing, "create_refund_receipt", fake_refund)

    def fake_stripe(pi, cents):
        stripe_calls.append((pi, cents)); return {"id": "re_1"}
    monkeypatch.setattr(stripe_pay, "refund", fake_stripe)

    monkeypatch.setattr(O, "get_order", lambda cx, oid: order)
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
    _patch_common(monkeypatch, invoice=None, order=order, refund_calls=refund_calls,
                  cust_calls=cust_calls, stripe_calls=stripe_calls)
    res = finance._refund_order_exec({"order_id": 8, "amount": 30.0}, {"cx": object()})
    assert stripe_calls == [("pi_123", 3000)]         # card refunded first
    assert len(refund_calls) == 1
    assert res["stripe_refund"] is True


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `doppler run --config dev -- python3 -m pytest tests/test_finance_refund_paid_only.py -v`
Expected: FAIL — the paid-only tests raise "invoice ... not found" (current code raises when `get_invoice` is None).

- [ ] **Step 3: Implement the fallback**

In `dashboard/finance.py:_refund_order_exec`, read the current code (~272-320). Replace the order/invoice resolution + the `inv = qb.get_invoice(...)` / `customer_id = ...` block (from the `invoice_id = params.get("invoice_id")` line through the `description = ...` assignment, i.e. the part BEFORE the Stripe-PI resolution) with:

```python
    invoice_id = params.get("invoice_id")
    order = None
    if params.get("order_id"):
        from dashboard.orders import get_order
        order = get_order(cx, int(params["order_id"]))
        if not order:
            raise ValueError(f"order #{params['order_id']} not found")
        if not invoice_id:
            invoice_id = order.get("external_ref")
    if not invoice_id:
        raise ValueError("invoice_id or order_id required")
    # Paid-only orders key external_ref = a checkout token (no QBO invoice); resolve
    # the order so we can fall back to email-based customer resolution.
    if order is None and cx is not None:
        try:
            from dashboard.orders import find_order_by_external_ref
            order = find_order_by_external_ref(cx, invoice_id)
        except Exception:
            order = None

    inv = qb.get_invoice(str(invoice_id))
    if inv:
        customer_id = (inv.get("CustomerRef") or {}).get("value")
        description = params.get("reason") or f"Refund for invoice {invoice_id}"
    elif order and (order.get("email") or "").strip():
        # Paid-only order: no QBO invoice. The customer that booked the SalesReceipt
        # is find_or_create_customer(email) (idempotent -> same customer). Record a
        # money-out RefundReceipt against it. Works even if qbo_sales_receipt_id is
        # the 'PENDING' sentinel (we only need the customer).
        cust = qb.find_or_create_customer(order.get("email"), order.get("name", ""))
        customer_id = cust.get("Id")
        _sr = order.get("qbo_sales_receipt_id")
        description = params.get("reason") or (
            f"Refund for order {invoice_id}"
            + (f" (SalesReceipt {_sr})" if _sr and _sr != "PENDING" else ""))
    else:
        raise ValueError(f"invoice {invoice_id} not found")
    if not customer_id:
        raise ValueError("could not resolve a QBO customer for this refund")
```

Leave everything AFTER this (the Stripe PaymentIntent resolution, the card refund, `create_refund_receipt(customer_id, amount, description=description)`, and the return dict) exactly as it is.

- [ ] **Step 4: Run tests to verify they pass**

Run: `doppler run --config dev -- python3 -m pytest tests/test_finance_refund_paid_only.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Regression — existing finance suites**

Run: `doppler run --config dev -- python3 -m pytest tests/test_bos_finance.py tests/test_finance_record_payment.py -v`
Expected: PASS (or any failure also present on `main`).

- [ ] **Step 6: Commit**

```bash
git add dashboard/finance.py tests/test_finance_refund_paid_only.py
git commit -m "feat(qbo): refund paid-only orders via RefundReceipt (Stage 2.5)"
```

---

## Full-suite gate (end of plan)

- [ ] Money-path regression (diff FAILED vs `main`):

Run: `doppler run --config dev -- python3 -m pytest tests/test_bos_finance.py tests/test_finance_record_payment.py tests/test_finance_refund_paid_only.py -v`
Expected: PASS, or any failure also present on `main`.

- [ ] Post-deploy manual (deployed env): refund a real paid-only biofield/begin order via the console; confirm the card is refunded (if a PI was on file) and a QBO **RefundReceipt** appears for the right customer/amount. Local QBO 400s are expected (`reference_qbo_local_token_stale`).
