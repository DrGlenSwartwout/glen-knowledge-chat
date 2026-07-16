# QBO Paid-Only Stage 3 — Remaining Retail + Recurring Flows — Design

**Date:** 2026-07-16
**Status:** Approved (design), pending spec review
**Depends on:** Stages 1 (#931) + 2 (#932) + 2.5 (#933) merged — the paid-only machinery exists: `orders.qbo_lines_json` + `qbo_sales_receipt_id` columns, `orders.set_order_qbo_lines` / `set_order_sales_receipt_id` / `claim_sales_receipt_slot`, and `qbo_sale.book_sale_on_payment` (idempotent atomic-claim, line-faithful).
**Owner:** Glen / RemedyMatch

## Problem

Six `create_invoice` sites in the retail + recurring flows still create unpaid
QBO invoices at/after checkout. Stage 3 converts them all to paid-only Sales
Receipts and retires the reconcile poller's forward role. **No new machinery** —
this reuses everything Stage 2 built.

**In scope (6 conversion targets + poller):**
- `_checkout_cart` (`app.py:25087`, source `reorder`, metadata kind `reorder`)
- `api_client_portal_checkout` (`app.py:20293`, source `portal-reorder`)
- `reorder_subscribe` (`app.py:25340`, metadata kind `subscribe`)
- `_ship_founding_reservation` (`app.py:25228`, off-session, kind `founding_ship`)
- `cron_charge_subscriptions` × 2 (`app.py:~32913` source `membership`, `~33011`
  source `subscription`) — off-session recurring
- `qbo_reconcile.py` — retire forward role (numeric-only filter)

**Out of scope (later):** wholesale + dropship (Stage 4 — discount-before-booking);
full removal of `qbo_reconcile.py` + `record_payment` (Stage 5, once no invoice
sources remain).

## Two conversion patterns

### Pattern I — interactive Stripe redirect (cart, portal-reorder, subscribe)
Mirror Stage 2's begin_checkout conversion exactly:
1. Generate a `checkout_ref` token at the top; drop `create_invoice`.
2. Persist the QBO line payload via `set_order_qbo_lines(cx, checkout_ref, {lines,
   discount_cents, tax_cents})`.
3. Re-key every former `inv.get("Id")` use to `checkout_ref`: `_ingest_order`
   external_ref, Stripe metadata `invoice_id` (keep the KEY name), the response
   `invoice_id`/empty `doc_number`/empty `customer_id`, and any
   `_record_referral_if_any(..., checkout_ref)`.
4. Book on payment in the `/begin/checkout-return` handler for that kind
   (`reorder`, `portal-reorder`, `subscribe`) — add a `book_sale_on_payment` call
   in the kind's block, resolving the order by `checkout_ref`, EXACTLY like the
   Stage-2 retail/biofield blocks.
5. **Preserve the cid-gated side-effects** (the Stage-2 trap): `set_order_stripe_pi`,
   `_settle_order_points`, `_settle_referral` must still fire for these kinds after
   `customer_id=""`. Wire them per-kind (mirror Stage 2's retail block) and pin with
   a spy test.
6. `reorder_subscribe` also writes the subscription row on return — that path keys
   off metadata, not the invoice, so it is unaffected; confirm it still runs.

### Pattern II — charge-then-book inline (founding_ship, cron ×2)
These charge a saved card off-session (`stripe_pay.charge_off_session`) and record
on success — no interactive return. Simpler:
1. Drop `create_invoice` (+ its `record_payment` in the cron).
2. Charge as today. On `res.status == "succeeded"`, `_ingest_order` (external_ref
   stays the Stripe charge id `res["id"]` — already not the invoice id, so no
   re-key needed there), then `set_order_qbo_lines(cx, <external_ref>, payload)` and
   `book_sale_on_payment(cx, order)` inline right after the charge.
3. One Sales Receipt per successful charge/period; the atomic claim + the
   per-charge unique `external_ref` (the Stripe charge id) make re-runs idempotent.
4. Preserve any settlement the cron did via the old invoice+payment path (read each
   site; the cron's own points/subscription bookkeeping is separate from the QBO
   write — keep it).

## Retire the reconcile poller's forward role
`qbo_reconcile.list_open_qbo_orders` selects open `reorder`/`portal-reorder` orders
with `external_ref GLOB '[0-9]*'` and polls each QBO invoice's balance. After
conversion these sources produce **token** external_refs (32-char hex), and a token
that happens to start with a digit would still be picked up and then error on
`get_invoice(token)`. Tighten the filter to match **only all-numeric, short**
external_refs (a real QBO invoice id, e.g. `NOT GLOB '*[^0-9]*' AND length < 20`) so
token-based paid-only orders are never polled. The poller keeps draining genuine
legacy in-flight invoices; it just ignores new paid-only orders. (Full removal is
Stage 5.)

## Testing
- Per flow: **guard test** — no `create_invoice` (Pattern I) / no
  `create_invoice`+`record_payment` (Pattern II) at checkout/charge; order keyed on
  the token (Pattern I) or the Stripe charge id (Pattern II); `qbo_lines_json`
  persisted.
- Pattern I: booking fires in the correct `/begin/checkout-return` kind block; the
  cid-gated **side-effects pinned** (spy test: `set_order_stripe_pi`,
  `_settle_order_points`, `_settle_referral` all fire for the converted kind); no
  double-book (idempotent claim).
- Pattern II: on a successful off-session charge exactly one Sales Receipt is
  booked; on a FAILED charge, none; re-running the cron for the same period books no
  second receipt (claim).
- `reorder_subscribe`: the subscription row still writes on return.
- Reconcile: `list_open_qbo_orders` returns legacy numeric-invoice orders and
  EXCLUDES token-based orders (add a token-external_ref row and assert it is not
  returned); existing `qbo_reconcile` tests still pass.
- Regression: begin/biofield paid-only (Stage 2) + finance suites green.

## Migration / mixed-state
- Zero backfill; go-forward only. Legacy in-flight `reorder`/`portal-reorder`
  invoices finish via the (now numeric-only) poller.
- After Stage 3, subscriptions are fully paid-only — month 1 AND recurring charges
  book Sales Receipts (the cron is converted here, per Glen), so no receipt/invoice
  mixed state for a subscription.
- Remaining invoice creators after Stage 3: wholesale + dropship (Stage 4) and the
  `qbo_test_invoice` diagnostic (intentional).

## Notes carried forward
- `_first_bank_account_id()` still has no ORDER BY (every Sales Receipt uses it);
  Stage-agnostic hardening tracked separately.
- Follow-ons still open: webhook-back the booking (closed-tab), stuck-`PENDING`
  reconciler sweep.
