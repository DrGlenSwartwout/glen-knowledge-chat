# QBO Paid-Only Stage 2.5 — Refunds for Paid-Only Orders — Design

**Date:** 2026-07-16
**Status:** Approved (design), pending spec review
**Depends on:** Stage 1 (#931) + Stage 2 (#932) merged — orders may now be paid-only (booked as a QBO SalesReceipt, `external_ref` = a `checkout_ref` token, no QBO invoice).
**Owner:** Glen / RemedyMatch

## Problem

`finance._refund_order_exec` (`dashboard/finance.py:272`) resolves the order's
QBO customer by looking up an **invoice**: `inv = qb.get_invoice(external_ref)` →
`customer_id = inv.CustomerRef.value`. Paid-only orders (biofield/begin, Stage 2)
have **no** QBO invoice — `external_ref` is a `checkout_ref` token — so
`get_invoice` returns None and the action raises `"invoice {token} not found"`.
Result: **you cannot refund a paid-only order from the console.** This worked
before Stage 2 (those orders had an invoice).

## Key facts

- The refund action is otherwise **already correct** for a paid-only sale: it
  refunds the card via Stripe (`stripe_pay.refund(pi, ...)`) using the order's
  `stripe_payment_intent` — which Stage 2 now stamps — and records a QBO
  **RefundReceipt** via `qbo_billing.create_refund_receipt(customer_id, amount,
  ...)`. A RefundReceipt is the correct QBO money-out object for refunding a
  SalesReceipt. Neither of those needs the invoice.
- The **only** thing the invoice lookup provides is `customer_id`.
- `find_or_create_customer(email, name)` is idempotent and returns the SAME QBO
  customer that booked the SalesReceipt (same email). So the customer can be
  resolved without any invoice or SalesReceipt read.

## Design

**One targeted change in `_refund_order_exec`.** Replace the hard dependency on a
QBO invoice for customer resolution with a fallback for paid-only orders:

1. Resolve the order dict up front when an `order_id` is given (existing), OR when
   only `invoice_id`/`external_ref` is given, additionally look it up via
   `find_order_by_external_ref(cx, external_ref)` so we have the order in hand.
2. Try `inv = qb.get_invoice(str(invoice_id))`:
   - **If an invoice exists** (legacy invoice orders): `customer_id =
     inv.CustomerRef.value` — unchanged path.
   - **If no invoice** AND the order is paid-only (`order["qbo_sales_receipt_id"]`
     is set, including the `'PENDING'` sentinel) OR the order simply has an email:
     resolve `customer_id = find_or_create_customer(order["email"],
     order.get("name",""))["Id"]`. Do NOT raise.
   - **If neither** (no invoice, no resolvable order/email): raise the existing
     `"invoice not found"` / not-found error.
3. Everything after customer resolution is UNCHANGED: resolve the Stripe PI from
   the order, card-refund first (only book QBO if real money went back), then
   `create_refund_receipt(customer_id, amount, description=...)`. The description
   for paid-only refunds includes the order ref and `qbo_sales_receipt_id` for
   QBO traceability (e.g. `"Refund for order <ref> (SalesReceipt <id>)"`).

**Stuck-`PENDING` orders refund fine** — customer resolution is by email, so an
order whose receipt id never finalized can still be refunded (card + RefundReceipt).

## Out of scope (confirmed with Glen)
- `_void_invoice_exec` stays invoice-only. Paid-only orders have no unpaid A/R
  invoice to void; the paid-only equivalent of "undo the sale" is a refund, covered
  above. Attempting to void a paid-only order still errors (correct — use refund).
- No change to how customers pay, to the Stripe refund mechanics, or to
  `create_refund_receipt` itself.
- Partial vs full refund unchanged (RefundReceipt is a single-line money-out for
  the requested amount).

## Testing
- **Paid-only refund books a RefundReceipt without an invoice:** an order with
  `external_ref = <token>`, no QBO invoice (`get_invoice` returns None), and a
  `qbo_sales_receipt_id` set → `_refund_order_exec` resolves the customer via
  `find_or_create_customer` and calls `create_refund_receipt` (assert it is called
  with the resolved customer_id + amount; assert it does NOT raise "invoice not
  found").
- **Card refund still fires** when the paid-only order has a `stripe_payment_intent`
  (assert `stripe_pay.refund` called before `create_refund_receipt`).
- **Legacy invoice path unchanged:** an order/invoice_id that DOES resolve via
  `get_invoice` still uses `inv.CustomerRef` (guard test — the new branch must not
  change the legacy behavior).
- **Stuck-`PENDING`:** `qbo_sales_receipt_id='PENDING'` still refunds (customer via
  email), no raise.
- **Genuinely unresolvable** (no invoice, no order, no email) still raises.
- Mutation check: the paid-only test fails on `main` (pre-Stage-2.5) with "invoice
  not found", passes after.

## Migration / mixed-state
None. Pure code change to the refund resolver; no schema, no backfill. Legacy
invoice orders and paid-only orders both refund correctly after this.
