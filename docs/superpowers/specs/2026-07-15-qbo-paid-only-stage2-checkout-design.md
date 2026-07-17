# QBO Paid-Only Stage 2 — Biofield + Begin Checkout — Design

**Date:** 2026-07-15
**Status:** Approved (design), pending spec review
**Depends on:** Stage 1 (merged, PR #931) — `qbo_billing.create_sales_receipt` exists.
**Owner:** Glen / RemedyMatch

## Problem

`biofield_checkout` (`app.py:8302`, source `biofield`) and `begin_checkout`
(`app.py:8371`, source `funnel`) both `create_invoice` at checkout — an unpaid
QBO A/R invoice created before any money moves. Payment is actually collected by
Stripe (card) or Zelle/Wise (alt-pay); the QBO invoice is only a record whose Id
is reused as the order join key and correlation ref. Stage 2 converts these two
routes to **paid-only**: no invoice at checkout, a **Sales Receipt** booked when
payment confirms.

**In scope:** `biofield_checkout`, `begin_checkout`, and `begin_checkout`'s
post-buy concierge upsell `/begin/concierge/add` (which today mutates the live
invoice). **Out of scope (later stages):** portal-reorder, `_checkout_cart`,
`reorder_subscribe`, founding-reservation, `cron_charge_subscriptions`,
wholesale/dropship, and retiring `qbo_reconcile.py`.

## Key facts established

- **Stripe/alt-pay already collect the money.** `_QBO_PAYMENTS_ACTIVE` is off; the
  QBO invoice is a record, not a payment surface, for both routes.
- **The invoice Id is a correlation key, not just the order key.** In
  `begin_checkout` `inv.get("Id")` also flows into `_record_referral_if_any`,
  `coupons.mark_redeemed(order_ref=...)`, `referrals.record_redemption(...)`, and
  the client response. All are opaque refs — a token substitutes cleanly.
- **The poller does NOT touch these routes.** `qbo_reconcile.QBO_SOURCES =
  ('reorder','portal-reorder')`; biofield/funnel are excluded. Stage 2 leaves
  reconcile alone.
- **Frontend contract (real, not cosmetic):** `static/begin-buy.html` reads
  `data.doc_number` (header text), `data.pay_link` (degraded card fallback), and
  `data.invoice_id` — which it stores as `cg.invoiceId` and **posts back** to
  `/begin/concierge/add` as `{slug, invoice_id}`. `biofield` uses `stripe_url`
  only.

## Design

### 1. Checkout token replaces the invoice Id
Generate one `checkout_ref` (e.g. `uuid4().hex`) at the top of each route, before
any QBO/pricing work. Use it everywhere `inv.get("Id")` flows today:
- `_ingest_order(external_ref=checkout_ref, ...)`
- Stripe metadata `invoice_id` → carries `checkout_ref` (keep the metadata KEY
  name so the return handler's existing `md.get("invoice_id")` read keeps working;
  its VALUE is now the token).
- `begin_checkout` correlation refs: `_record_referral_if_any(..., checkout_ref)`,
  `coupons.mark_redeemed(order_ref=checkout_ref)`,
  `referrals.record_redemption(..., checkout_ref)`.
- Client response `invoice_id` field → carries `checkout_ref` (keep the field NAME
  for frontend compat; the frontend just needs a stable id to post to
  `/begin/concierge/add`).

### 2. Two new order columns (`dashboard/orders.py`)
- `qbo_sales_receipt_id TEXT` — idempotency marker; booking no-ops if set.
- `qbo_lines_json TEXT` — the exact QBO line payload captured at checkout:
  `{"lines": [...], "discount_cents": N, "tax_cents": N}`. Lets the paid handler
  book a **line-faithful** Sales Receipt (Glen: line-faithful, not a lossy single
  "total" line). Added via the module's existing `ALTER TABLE orders ADD COLUMN`
  migration list.

### 3. `book_sale_on_payment(cx, order)` — the one booking helper
New helper (in `dashboard/qbo_sale.py`, a small new module, or alongside the
existing membership booking in `app.py` — the plan picks the home that keeps
layering clean). Contract:
- Idempotent: if `order["qbo_sales_receipt_id"]` is set, return without booking.
- Resolve the QBO customer via `qbo_billing.find_or_create_customer(email, name)`.
- Rebuild lines from `qbo_lines_json`; call
  `qbo_billing.create_sales_receipt(cust, lines, discount_cents=..., tax_cents=...,
  email_to=email)`.
- Persist the returned receipt Id onto the order (`qbo_sales_receipt_id`).
- Best-effort: never raises into the payment path (mirrors `_book_membership_qbo`).

### 4. Wire booking into both paid transitions
`begin_checkout` supports card AND Zelle/Wise, so both matter:
- **Stripe return handler** (`app.py:~9358`): for `kind in ("biofield","funnel")`,
  replace the `qbo_billing.record_payment(cid, amount, inv)` (apply-to-invoice)
  call with `book_sale_on_payment(cx, order)` (order resolved by `checkout_ref`).
- **Alt-pay confirm / `set_order_payment`**: book there too. Idempotency (§3)
  guarantees exactly one Sales Receipt regardless of which path fires first.

### 5. Drop `create_invoice` from both routes
Remove the `qb.create_invoice(...)` call and the `allow_online_pay` /
`_QBO_PAYMENTS_ACTIVE` branch from `biofield_checkout` and `begin_checkout`. No
QBO write happens at checkout.

### 6. Convert `/begin/concierge/add` to append-to-order
Today it calls `qbo_billing.add_invoice_line` on the live unpaid invoice. Paid-only
has no invoice yet. Rework it to:
- Resolve the pending order by `checkout_ref` (posted as `invoice_id`).
- Append the added item to the order's `qbo_lines_json` and re-total the order
  (`total_cents`, items). NO QBO call pre-payment.
- The Sales Receipt booked at payment (§3–4) then includes the added line
  automatically. (This only reaches the customer on the alt-pay / pay-later path;
  card checkout redirects to Stripe before the concierge screen shows.)

### 7. Response-shape compatibility (spec-time verification — Glen approved)
- `invoice_id` (response + Stripe metadata): keep the KEY, value = `checkout_ref`.
- `doc_number`: no QBO doc number exists pre-payment. Return `""` (or the order's
  short id) and update `begin-buy.html` line 745 so the header reads sensibly
  (e.g. "Order received" instead of "Invoice #<blank> created").
- `pay_link`: `_QBO_PAYMENTS_ACTIVE` is off, so `get_invoice_pay_link` is already
  inert; drop it from the response and remove the dead `pay_link` fallback branch
  in `begin-buy.html` (~line 751), leaving the existing `payment_error` message.
- The plan MUST verify these three against `begin-buy.html` before changing them,
  and keep card (`stripe_url`) and alt-pay (`/invoice/<token>`) paths working.

## Testing
- **Guard (mutation-style):** assert NO `POST /invoice` at checkout for biofield
  and begin — inject the old `create_invoice`, watch the guard go red.
- `book_sale_on_payment`: line-faithful receipt built from `qbo_lines_json`;
  idempotency (double-fire → one receipt); best-effort never-raise.
- Re-key: order created with `external_ref == checkout_ref`; referral/coupon/gift
  refs carry the token; response `invoice_id == checkout_ref`.
- Stripe-return path books exactly one Sales Receipt for kind biofield and funnel;
  alt-pay `set_order_payment` books one; both together still one (idempotent).
- `/begin/concierge/add`: appends to `qbo_lines_json` + re-totals, makes NO QBO
  call; the subsequent booked receipt includes the added line.
- Regression: existing begin/biofield checkout, coupon, and referral suites pass.

## Migration / mixed-state
- Zero backfill. Pre-existing biofield/funnel orders that already have a QBO
  invoice keep it (they finish their current lifecycle; `record_payment` path
  stays available for in-flight ones). Cutover is go-forward only.
- New columns default NULL; existing rows unaffected.

## Out of scope (restated)
Portal-reorder / cart / reorder-subscribe / founding-reservation /
subscription-cron / wholesale / dropship conversions; retiring `qbo_reconcile.py`
and `record_payment`. Each is a later stage.
