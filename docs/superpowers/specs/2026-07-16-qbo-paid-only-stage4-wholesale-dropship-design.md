# QBO Paid-Only Stage 4 — Wholesale + Dropship + Dispensary — Design

**Date:** 2026-07-16
**Status:** Approved (design), pending spec review
**Depends on:** Stages 1/2/2.5/3 + pricing + ship-credit fix all merged. Machinery exists: `orders.set_order_qbo_lines`/`set_order_sales_receipt_id`/`claim_sales_receipt_slot`, `qbo_sale.book_sale_on_payment`, the return-handler booking block.
**Owner:** Glen / RemedyMatch

## Problem

The last three `create_invoice` sites (excluding the `qbo_test_invoice` diagnostic)
are the practitioner flows in `dashboard/wholesale_checkout.py` and
`dashboard/dropship_checkout.py`. They create unpaid QBO invoices; collection is
via Stripe (card) or Zelle/Wise (alt-pay), so the invoice is only a record. Convert
all three to paid-only Sales Receipts. Paid-only holds for practitioners too (no
NET-terms A/R — Glen, Stage 1).

**In scope (3 flows):**
- `wholesale_checkout.build_order` — wholesale (metadata `kind="wholesale"`)
- `dropship_checkout.build_dropship_order` — dropship (metadata `kind="wholesale"`)
- `dropship_checkout.build_client_order` — dispensary→patient (metadata `kind="client"`)

Also `wholesale_checkout.build_module_order` (cert module) if it shares the
create_invoice+apply_invoice_discount shape — convert it the same way.

**Out of scope:** the `qbo_test_invoice` diagnostic (intentional). Retiring
`qbo_reconcile.py` + `record_payment` = Stage 5.

## The key complexity: discount resolved BEFORE booking

`build_order` / `build_dropship_order` today:
```
inv = create_invoice(cust, lines)
redeemed = wallet.redeem_for_order(pid, subtotal, invoice_id)   # commits the redemption, keyed on invoice_id
if redeemed: inv = apply_invoice_discount(invoice_id, redeemed) # discount AFTER creation
```
A Sales Receipt is final (no post-hoc discount line edit like `apply_invoice_discount`),
so resolve the redeem FIRST, keyed on the `checkout_ref` token, then bake it into the
booked receipt.

`build_client_order` (dispensary) already passes `discount_cents=redeem+ship_credit`
at `create_invoice` time — so it needs only the token re-key + drop-create_invoice +
persist-payload + book, no reorder of the discount.

## Design

For each flow:
1. **Generate `checkout_ref = uuid4().hex`** before any QBO/wallet work.
2. **Resolve the discount up front, keyed on the token:**
   - wholesale/dropship: `redeemed = wallet.redeem_for_order(pid, subtotal, checkout_ref)`
     (and `wallet.redeem_for_module(...)` for `build_module_order`) BEFORE building the
     payload. The fee-free earn (`wallet.earn_fee_free(pid, charged, checkout_ref)`) and
     `_pp.record_order(pid, invoice_id=checkout_ref)` also re-key invoice_id → token.
   - dispensary: keep its existing pre-creation `discount_cents = redeem + ship_credit`.
3. **Drop `create_invoice` and `apply_invoice_discount`.** Persist the payload:
   `set_order_qbo_lines(cx, checkout_ref, {"lines": <lines>, "discount_cents": <redeemed
   (+ship_credit for dispensary)>, "tax_cents": 0})`. GET stays absorbed — `get_cents`
   recorded on the order (`_ingest_order(get_cents=...)`), never charged, never a receipt
   tax line.
4. **Re-key the order + response:** `_ingest_order(source=<wholesale|dropship|dispensary>,
   external_ref=checkout_ref, ...)`; the returned `out` dict keeps the field name
   `invoice_id` with the token as value, `customer_id=""`, `doc_number=""`. The Stripe
   metadata (`_stripe_checkout_url_for_order`) carries the token under `invoice_id`.
5. **Charge basis:** = `subtotal − redeemed` (+ shipping if the quote has it), which is
   exactly `out["total"]` today (the discounted invoice total) — so the card charge is
   unchanged; just compute it from the pre-resolved redeem instead of the invoice
   TotalAmt. GET absorbed (not in the charge). Charge == booked Sales Receipt.
6. **Book on payment:**
   - Stripe-return (card): add `"wholesale"` and `"client"` to the return-handler
     booking-block kind-list (the `if pi_id and _kind in (...)` gate) so
     `book_sale_on_payment` fires for the resolved order (idempotent atomic claim).
     Confirm the `record_payment` invoice-apply guard still skips these (they carry
     `customer_id=""` → `cid` falsy).
   - Alt-pay (Zelle/Wise): the operator-confirm path (`orders.record_payment` →
     `_record_payment_exec`) already calls `book_sale_on_payment` — a no-op until
     `qbo_lines_json` exists, which it now does. One Sales Receipt either way.

## Testing
- Per flow: guard — no `create_invoice`/`apply_invoice_discount` at checkout; order
  keyed on the token; `qbo_lines_json` persisted with `discount_cents == redeemed`.
- **Discount fidelity:** a wholesale order with a Wellness-Credit redemption books a
  Sales Receipt whose total == `subtotal − redeemed` == the amount charged; the
  redeem is committed once (keyed on the token, idempotent).
- Fee-free earn + `_pp.record_order` are keyed on the token (not a QBO invoice id).
- `get_cents` recorded on the order (GET absorbed, remittance).
- Booking: card path books exactly one Sales Receipt via the return handler; alt-pay
  path books one via `_record_payment_exec`; both together still one (claim). No
  double-book; side-effects (where applicable) preserved.
- Dispensary (`build_client_order`): points redeem + ship-credit discount preserved,
  charge == receipt.
- Regression: `tests/test_wholesale_checkout.py`, `tests/test_dropship_checkout.py`,
  `tests/test_dispensary_*`, wallet suites.

## Migration / mixed-state
Zero backfill; go-forward. Legacy in-flight wholesale/dropship invoices finish via
the (numeric-only) reconcile poller. After Stage 4, the only remaining
`create_invoice` caller is the `qbo_test_invoice` diagnostic → Stage 5 can retire the
poller + `record_payment`.

## Notes
- The wallet redeem is committed at checkout (pre-payment) today; paid-only keeps that
  timing (an abandoned checkout still debits the wallet — pre-existing, unchanged).
- Ship-credit: dispensary applies it at creation (already in the charge via
  `discount_cents`); confirm wholesale/dropship don't separately apply `_plan_ship_credit`
  (they use the wallet, not ship_credit) — no charge/receipt gap to re-open.
