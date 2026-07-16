# QBO Paid-Only (Sales Receipts) — Design

**Date:** 2026-07-15
**Status:** Approved (design), pending spec review
**Owner:** Glen / RemedyMatch

## Problem

Every order flow writes a QBO **Invoice** at checkout, *before* any money moves
(`dashboard/qbo_billing.py:create_invoice`, ~13 call sites). The invoice lands in
QBO Accounts Receivable immediately and sits **unpaid** until `qbo_reconcile.py`
polls QBO for a zero balance and flips the board. This produces phantom
receivables in QBO for orders that may never pay, and recognizes sales-tax
liability / income on unpaid orders.

**Decision:** No unpaid invoice ever hits QBO — **paid-only, everywhere**
(retail, membership, wholesale, dropship/practitioner). Every sale is written to
QBO as a **Sales Receipt** at the moment payment confirms.

## Key facts that make this feasible

- **Collection is already decoupled from QBO.** `/api/invoice/<token>/pay`
  (`app.py:38926`) collects card via **Stripe** and Zelle/Wise via alt-pay. It
  does *not* use the QBO invoice's hosted pay link. `_QBO_PAYMENTS_ACTIVE` is
  off — QBO never collected card for these flows. So changing *when* we write to
  QBO does not change *how* customers pay.
- **`order_id` already travels in Stripe metadata** (`app.py:38950`), so the
  Stripe-success → order join no longer needs the QBO invoice Id.
- A Sales Receipt is born paid: it books income + the deposit/undeposited-funds
  account and **never touches A/R**.

## Design

### 1. New QBO write path — `create_sales_receipt`
Add `create_sales_receipt(customer, lines, *, discount_cents=0, tax_cents=0,
email_to=None)` to `dashboard/qbo_billing.py`. Reuse `_build_invoice_lines` and
the same `TxnTaxDetail`/`GlobalTaxCalculation` override so discount and
app-computed sales-tax parity with `create_invoice` is exact. Body posts to
`POST /salesreceipt`. No `AllowOnlineCreditCardPayment`/`AllowOnlineACHPayment`
(there is nothing to pay — it is already paid).

### 2. Move the QBO write from checkout → payment confirmation
The ~13 `create_invoice` call sites stop writing to QBO at checkout. Checkout
still creates the **local** order (board row) and collects payment exactly as
today. A single helper `_book_sale_qbo(order)` performs the QBO write and fires
at the two real choke points where an order flips paid:

1. **Stripe return handler** — `app.py:~9358` (currently the in-house paid
   branch and the `record_payment` branch).
2. **Alt-pay / manual mark-paid** — `dashboard/orders.py:set_order_payment`
   (Zelle/Wise confirmed, operator "mark paid").

`_book_sale_qbo` resolves the QBO customer (`find_or_create_customer`), builds
lines from the order's stored line items, and calls `create_sales_receipt`.

### 3. Idempotency
Store the returned Sales Receipt Id in a dedicated order column
(`qbo_sales_receipt_id`). `_book_sale_qbo` no-ops if that column is already set.
This makes it safe against Stripe webhook retries, double mark-paid, and any
reconcile re-run. (Mutation-tested: double-fire → exactly one receipt.)

### 4. Re-key the order↔payment join
Orders no longer use the QBO invoice Id as `external_ref`. At checkout,
`external_ref` = the order's own token/id. Stripe metadata already carries
`order_id`, so the Stripe-return path routes by `order_id` (replacing
`find_order_by_external_ref` for new orders). The QBO Sales Receipt Id lives in
`qbo_sales_receipt_id`, never in `external_ref`.

### 5. Retire the QBO-polling reconcile (go-forward)
`qbo_reconcile.py` exists only because unpaid QBO invoices were paid out-of-band
and the board never flipped. With paid-only, the **payment rail** flips the board
and books QBO in the same step — QBO no longer leads. The poller stays wired for
**pre-existing** open invoices already in QBO but receives **no new orders**
(`QBO_SOURCES` orders stop being produced). It can be removed entirely once the
legacy open invoices are closed.

### 6. Mixed-state / migration
**Zero backfill.** Every QBO invoice already created stays as-is in QBO;
`qbo_reconcile.py` keeps closing them. Cutover is purely go-forward: after
deploy, new orders produce Sales Receipts; in-flight invoices finish their
existing lifecycle. `qbo_billing.record_payment` (apply-payment-to-invoice) is
**kept** solely for those legacy in-flight invoices, not called for new orders.

### 7. Dead flags
`_QBO_PAYMENTS_ACTIVE` and `allow_online_pay` become inert for new orders (no
invoice to attach a hosted card link to). Leave `_QBO_PAYMENTS_ACTIVE` defined
for the legacy path; drop `allow_online_pay` from all new-order QBO calls.

## Call-site inventory (to convert in the plan)
`create_invoice` callers that move to paid-only Sales-Receipt-on-payment:
- `app.py` begin checkout (`~8319`, `~8413`), memberships `_book_membership_qbo`
  (`~9079/9088/9237`), and console/test paths (`~24966`, `~25107`, `~25219`,
  `~32792`, `~32890`), local pay (`~20172`).
- `dashboard/dropship_checkout.py:104`, `:285`
- `dashboard/wholesale_checkout.py:54`, `:87`
- Diagnostics/test-invoice routes (`app.py:5046 qbo_test_invoice`) — keep as a
  QBO write-layer smoke test but point at `create_sales_receipt` (or keep both).

Each site's exact conversion (where the paid-hook lands for that flow) is worked
out in the implementation plan.

## Testing
- `create_sales_receipt` body-shape + discount + tax parity vs the existing
  `create_invoice` tests (mirror `tests/test_invoice_srp_anchor.py` /
  `test_ff_add_to_invoice.py` patterns for the salesreceipt body).
- `_book_sale_qbo` idempotency: double-fire → one receipt (assert second call
  no-ops on `qbo_sales_receipt_id`).
- Stripe-return path and `set_order_payment` path each book exactly one receipt.
- **Guard test (mutation-style):** assert **no** `POST /invoice` happens at
  checkout for retail, wholesale, dropship, and membership flows — inject the old
  behavior, watch it go red.
- Legacy path untouched: `qbo_reconcile` tests still pass; `record_payment`
  still applies to a pre-existing invoice.

## Out of scope
- No change to how customers pay (Stripe / Zelle / Wise unchanged).
- No backfill of historical invoices into Sales Receipts.
- No removal of `qbo_reconcile.py` in this change (deferred until legacy open
  invoices drain).
- NET-terms / true A/R invoicing for practitioners — explicitly **not** offered
  (Glen chose paid-only everywhere).
