# QBO Paid-Only — Auto-heal Stuck-PENDING Sales-Receipt Bookings — Design

**Date:** 2026-07-16
**Status:** Approved (design), pending spec review
**Depends on:** QBO paid-only migration complete. `qbo_sale.book_sale_on_payment` (atomic claim `qbo_sales_receipt_id` NULL→'PENDING'→real), `qbo_billing.create_sales_receipt`, `orders.set_order_sales_receipt_id`.
**Owner:** Glen / RemedyMatch

## Problem

`book_sale_on_payment` claims the booking slot (`qbo_sales_receipt_id` NULL→`'PENDING'`), writes the QBO Sales Receipt, then stamps the real id (`'PENDING'`→id). A crash/failure in the tiny window between the claim and the stamp leaves the order stuck at `'PENDING'` forever — the order is marked paid, but `book_sale_on_payment` returns early on `'PENDING'`, so it never re-books. Two cases:
- **Case A:** claim OK, `create_sales_receipt` failed → stuck PENDING, **no receipt in QBO** (revenue missing).
- **Case B:** claim + receipt OK, stamp write failed → stuck PENDING, **receipt exists in QBO**.

A naive auto-rebook would double-book Case B. Prod currently has **0 stuck-PENDING orders**, so this is purely going-forward. Rare, but currently invisible + unrecoverable.

## Design (automated auto-heal, exact-match — Glen chose)

### Part 1 — self-identifying receipts (makes matching exact, not fuzzy)
`create_sales_receipt` gains an optional `private_note` param and stamps it into the QBO SalesReceipt `PrivateNote`. `book_sale_on_payment` passes `private_note = f"order:{order['external_ref']}"` (the `checkout_ref` token). Every receipt is now linkable back to its exact order. (Existing pre-change receipts lack it — but there are 0 stuck-PENDING orders, so no legacy to mis-heal.)

### Part 2 — the heal sweep (`dashboard/qbo_heal.py`, new)
`heal_pending_receipts(cx, *, get_receipt_by_ref, create/stamp deps, older_than_min=10)`:
1. Select orders WHERE `qbo_sales_receipt_id = 'PENDING'` AND `updated_at < now - older_than_min` (the age guard avoids catching an in-flight booking).
2. For each, look up the QBO SalesReceipt carrying `order:<token>` in PrivateNote:
   - **Found (Case B):** `set_order_sales_receipt_id(cx, order_id, found_id)` — stamp, done.
   - **Not found (Case A):** clear `qbo_sales_receipt_id` → NULL, then `book_sale_on_payment(cx, order)` — re-claims + re-books (now with the token in PrivateNote), stamps.
3. Best-effort per order (one bad order logged + skipped, never aborts the sweep). Idempotent + exact — since all future receipts carry the token and there are 0 legacy PENDING orders, it can never double-book.

**QBO lookup (the one real risk):** prefer a query `SELECT * FROM SalesReceipt WHERE PrivateNote LIKE '%order:<token>%'` if QBO supports LIKE on PrivateNote; **the plan MUST verify this against live QBO first**. Fallback if unsupported: resolve the order's QBO customer (`find_or_create_customer(email)`), query that customer's recent SalesReceipts (`CustomerRef` + a TxnDate window around the order), and scan PrivateNote client-side for the token. Either way the match is by the exact token, never fuzzy amount-only.

### Part 3 — the cron endpoint + fold-in
- New route `POST /api/cron/qbo-heal-pending` (cron/console-secret gated, same auth as the other `/api/cron/*` / reconcile route used) → runs `heal_pending_receipts` and returns `{ok, healed: [...], count}`.
- **Fold the curl into an existing daily cron** (no new Render service — Glen's call): add one curl of `/api/cron/qbo-heal-pending` to an already-active daily cron script (`scripts/run_briefings_cron.py` is the proposed host; it already curls the web service daily). The plan confirms the exact host script + that it's active in `render.yaml`.

## Testing
- **Part 1:** `create_sales_receipt` stamps `PrivateNote` when `private_note` given (body-shape test); `book_sale_on_payment` passes `order:<token>`.
- **Case A (offline, mocked):** stuck-PENDING order, QBO lookup returns no receipt → cleared + re-booked → exactly one receipt, `qbo_sales_receipt_id` now real.
- **Case B (offline, mocked):** QBO lookup returns the receipt (token match) → order stamped with that id, `create_sales_receipt` NOT called (no double-book).
- **Age guard:** a PENDING order updated < N min ago (in-flight) is NOT swept.
- **Best-effort:** one order whose lookup raises is skipped; the sweep continues and returns the rest.
- **Endpoint:** `POST /api/cron/qbo-heal-pending` gated (401 without secret); runs the sweep.
- **Live-QBO verify (plan step, deployed env):** confirm the PrivateNote query/lookup actually returns the receipt from real QBO (units can't prove QBO's query semantics — `feedback_verify_against_live_api`).

## Out of scope
Full per-kind settlement parity on closed tab (points/referral) — separate follow-on. This heals the RECEIPT (revenue integrity) only.
