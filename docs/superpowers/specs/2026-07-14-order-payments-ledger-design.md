# Order Payments Ledger — Design Spec

**Date:** 2026-07-14
**Author:** Glen Swartwout + Claude
**Status:** Approved for planning

## Problem

An order in the console stores exactly one payment: a single `pay_method` +
amount summary field. A real order can be settled by several payments from
different sources (e.g. Dana Tamraz: Stripe card $222.91 + Zelle $131 + a third
Zelle $58.91 against a $412.82 order). Today that is impossible to record or
display in the app; the only place multiple partial payments live is
QuickBooks. There is also no way to correct a mis-keyed payment. This caused a
real double-count (a manual $221 proxy payment keyed on a sibling invoice while
Stripe had already recorded the real $222.91 on another).

## Goal

Add a first-class, correctable, multi-payment ledger per order that is the
app's source of truth and stays in sync with QuickBooks, shown on the console
order card, the client-facing invoice, and the orders/money boards.

## Decisions (locked during brainstorming)

1. **Source of truth:** app ledger is authoritative; each mutation syncs to QBO.
2. **Corrections:** void + re-add (immutable audit trail); payments are never
   edited in place. A wrong payment is marked `void` (kept visible with a
   reason) and a corrected one is added.
3. **Surfaces:** console order/invoice card, client-facing invoice page, and
   orders board + `/console/money` payments view.
4. **Sync architecture (Approach C):** synchronous push to QBO on every
   mutation; on failure the row is flagged `pending`/`error` and shown with a
   "needs resync" badge; a one-click **idempotent** resync repairs it. No cron.
5. **Stripe auto-payment:** `checkout-return` stops writing QBO directly and
   instead adds a ledger row (`source='stripe'`), which owns the single QBO
   push. Idempotent on the Stripe PaymentIntent id.
6. **Backfill of all historical orders:** deferred to a **later slice**. This
   build creates the table and handles new + Dana orders only.
7. **Client visibility:** the client invoice shows only `active` rows; voided
   rows are hidden from the client (visible in the console).

## Data model — `order_payments` (in `chat_log.db` / `LOG_DB`)

Balance is always derived, never stored.

| Column | Type | Purpose |
|---|---|---|
| `id` | INTEGER PK | |
| `order_id` | INTEGER | links to the order |
| `amount_cents` | INTEGER | positive |
| `method` | TEXT | existing dropdown set: Credit card (Stripe), eProcessing, Check, Cash, Venmo, PayPal, Zelle, Wise |
| `source` | TEXT | `stripe` (auto) · `manual` (hand-keyed) · `legacy` (future backfill) |
| `external_ref` | TEXT | Stripe PI id or Zelle/check memo; idempotency key for auto rows (nullable) |
| `paid_at` | TEXT | ISO date, editable (backdated Zelle records on the correct day) |
| `note` | TEXT | optional |
| `status` | TEXT | `active` · `void` |
| `void_reason` | TEXT | filled on void |
| `voided_at` | TEXT | filled on void |
| `qbo_payment_id` | TEXT | QBO Payment.Id this row created; the idempotency anchor |
| `qbo_sync` | TEXT | `synced` · `pending` · `error` |
| `created_at` | TEXT | |
| `updated_at` | TEXT | |
| `created_by` | TEXT | actor |

**Balance** = order invoice total − SUM(`amount_cents` WHERE `status='active'`).
Negative balance is displayed as a credit / overpayment.

## Backend — `dashboard/order_payments.py` (pure functions)

- `list_payments(cx, order_id)` → rows (active + void), newest first
- `add_payment(cx, order_id, amount_cents, method, source, external_ref, paid_at, note, actor)`
  → inserts row, then pushes one QBO payment via `qbo_billing.record_payment`
  (store returned Id in `qbo_payment_id`, set `qbo_sync='synced'`); on exception
  set `qbo_sync='error'` and keep the app row. Idempotent when `external_ref`
  matches an existing non-void row of the same order (returns the existing row).
- `void_payment(cx, payment_id, reason, actor)` → set `status='void'`; call QBO
  void/delete on `qbo_payment_id`. **No-op against QBO if `qbo_payment_id` is
  null.** Idempotent (voiding an already-void row is a no-op).
- `balance(cx, order_id)` → `{invoice_cents, paid_cents, balance_cents}`
- `resync(cx, payment_id)` → for `pending`/`error` rows, re-attempt the QBO
  push (or void) using `qbo_payment_id` presence to avoid double-posting.

New helper in `dashboard/qbo_billing.py`: `void_payment(qbo_payment_id)`
(delete/void a QBO Payment), mirroring the existing `record_payment`.

## Routes (console-auth, MONEY_SEND / OWNER·OPS — same tier as `finance.record_payment`)

- `GET  /api/orders/<oid>/payments` → `{rows, balance}`
- `POST /api/orders/<oid>/payments` → add `{amount, method, paid_at?, external_ref?, note?}`
- `POST /api/orders/payments/<pid>/void` → `{reason}`
- `POST /api/orders/payments/<pid>/resync` → idempotent QBO repair

## Stripe integration (fixes the Dana double-count)

Where `checkout-return` currently records a QBO payment for the captured Stripe
charge, replace that with `order_payments.add_payment(source='stripe',
method='Credit card (Stripe)', external_ref=<PaymentIntent id>, ...)`. The
ledger owns the single QBO push. Because `add_payment` is idempotent on the PI
id, a re-hit of `checkout-return` (retry, refresh) can never create a duplicate
payment. `checkout-return` no longer calls QBO directly.

## UI

### Console order card (`static/order-new.html`, edit mode)
Replace the single method/amount line with a **payments panel**:
- table: date · method · amount · source badge · status (voided rows struck
  through with reason)
- **Paid / Balance** running line (credit shown if overpaid)
- **Add payment** mini-form: amount, method dropdown, date, ref, note
- per-row **Void** (prompts for reason)
- `pending`/`error` rows show a "needs resync" badge + **Resync** button

The order's legacy `pay_method` field remains for back-compat but the panel is
the real record. `pay_status` derives: `unpaid` (0 paid) · `partial`
(0 < paid < total) · `paid` (balance ≤ 0).

### Client invoice (`/invoice/<token>`)
Read-only "Payments received" list (active rows only) + "Balance due".

### Orders board / money (`/console/orders`, `/console/money`)
Paid-vs-balance per order. `/api/payments` extended to union manual
`order_payments` with the existing Stripe ledger so Zelle/check show alongside
card charges (today it is Stripe-only).

## Migration

- `CREATE TABLE IF NOT EXISTS order_payments (...)` — idempotent, runs in the
  web container where `/data/chat_log.db` is mounted.
- **No** bulk historical backfill in this slice (deferred).
- Dana's order is hand-corrected to her three real rows through the new
  Add-payment endpoint after ship.

## Testing (TDD, run under `doppler -c dev pytest`)

Unit (`tests/test_order_payments.py`):
- add → balance decreases; overpayment → negative balance (credit)
- same PI id added twice → exactly one row (idempotency)
- void → excluded from balance; voiding twice is a no-op
- void with null `qbo_payment_id` → no QBO call, still marks void
- resync repairs a `pending` row without double-posting (uses `qbo_payment_id`)

Route (`tests/test_order_payments_routes.py`):
- auth: non-OWNER/OPS rejected
- `checkout-return` with a captured PI creates exactly one ledger row; a second
  call creates none

Honor the `PYTEST_CURRENT_TEST` email guard so the suite does not send live
mail.

## Out of scope (future slices)

- Bulk backfill of all historical orders' single `pay_method` into `legacy`
  rows (provenance-neutral, no new QBO posts).
- Refund handling (negative payments) — separate concern.
- Payment-plan / scheduled-installment automation.
