# Settlement Durability — Close the Crash-Strand Window + Non-Atomic Points Race — Design

**Date:** 2026-07-17
**Status:** Approved (design), pending spec review
**Depends on:** Per-kind settlement parity (PR #953). `dashboard/order_settlement.settle_paid_order_effects`, the redirect (`begin_checkout_return`) and webhook (`webhook_stripe`) dispatch sites, `dashboard/points.py`, `dashboard/orders.py`.
**Owner:** Glen / RemedyMatch

## Problem

The #953 whole-branch review surfaced two residual issues in per-kind settlement:

- **I1 — crash-strand.** The webhook runs settlement only inside its `not qbo_sales_receipt_id` guard (i.e. only when the receipt isn't booked yet), and it books the receipt *before* settling in that same block. A hard crash (deploy restart, OOM, gunicorn timeout) between booking and settling strands settlement: on any later delivery the receipt is already booked, so the guard is false and settlement never runs again. The highest-value stranded effect is the recurring `subscribe` row (silently no autoship).
- **I2 — non-atomic points.** `points.has_entry` is a check-then-insert guard (`points.py`), safe only within one process (single shared sqlite connection serializes it). Under prod's gunicorn 2-worker topology, a concurrent redirect+webhook settle of the same order can both pass `has_entry` and both insert → double-earn/double-redeem. Referral *cash* (`rewards.accrue_cash`) and the subscription row (`subscriptions.create_once`) are already atomic; only the **points** path is not.

## Key enabling fact (shaped the design)

The settlers require the full Stripe session metadata (`md`: `kind`, `slug`, `cadence_months`, `stash_key`, `items`, `ship`, `grant_group_months`, `patient_email`, `subtotal_cents`, `email`, …) and the session id (`sid`). The `orders` row persists **neither** — only `external_ref` (= `invoice_id` = `order_ref`) and `stripe_payment_intent` (= `pi_id`). So a **heal-style backfill sweep driven off stored orders cannot re-run the kind-specific settlers** (no way to recover `md`/`sid`).

**However:** the Stripe webhook always arrives carrying the full live session, and Stripe **redelivers** any event whose handler does not return 200 (`webhook_stripe` returns 500 on an unhandled exception; a hard crash returns nothing). So the webhook is the universal settlement backfill — no metadata persistence and no heal cron are needed. This is why the design lives entirely in the two request paths + the points table.

## Design

### Part 1 — `settled_at` marker (`dashboard/orders.py`)
Add a nullable `settled_at TEXT` column via the existing idempotent `ALTER TABLE orders ADD COLUMN` migration tuple in `init_orders_table`. Add `mark_order_settled(cx, order_id)` (sets `settled_at = _now()` where currently NULL) and expose `settled_at` on the order dict reads. The marker means "per-kind settlement has been attempted for this order."

### Part 2 — decouple webhook settlement from the receipt-booked guard (closes I1)
In `webhook_stripe` (`checkout.session.completed`), move the `settle_paid_order_effects(...)` call OUT of the `if _wo and _wo["qbo_lines_json"] and not _wo["qbo_sales_receipt_id"]:` booking guard. Instead, after the booking block, run settlement whenever the order is paid-only (`qbo_lines_json` present) AND `settled_at IS NULL` — regardless of whether the receipt is already booked. Mark settled after. Keep it best-effort (own try/except; a settler exception must not 500 the webhook or block booking).

Effect:
- **Hard crash between book and settle** → `settled_at` stays NULL, handler returns no 200 → Stripe redelivers → retry (with live session) runs settlement → marks settled. Closed.
- **Redirect crash after booking, before settling** → the webhook for that order (always delivered) sees `settled_at IS NULL` and settles. Closed.
- **Already settled** (redirect or a prior webhook completed) → `settled_at` set → webhook skips settlement (idempotent no-op either way).

### Part 3 — redirect marks settled (`begin_checkout_return`)
After the redirect's `settle_paid_order_effects(...)` dispatch, call `mark_order_settled`. Optionally skip the dispatch when `settled_at` is already set (minor; settlers are idempotent regardless). No behavior change to what the redirect settles.

### Part 4 — atomic points (closes I2) (`dashboard/points.py`)
Convert the check-then-insert guard into an atomic one, mirroring `subscriptions.create_once` / `rewards.accrue_cash`:
- Add a UNIQUE index `points_ledger(order_ref, reason, scope)` in `init_points_table` (idempotent try/except ALTER-style).
- Change the ledger insert (`_add`) to `INSERT OR IGNORE` and treat "row already present" as the idempotent no-op (return the existing balance / signal skip). `has_entry` may stay as a cheap fast-path pre-check, but correctness now rests on the DB-level UNIQUE constraint, so a cross-process concurrent double converges to one row.

All existing write paths already treat `(order_ref, reason, scope)` as the idempotency key (buyer `earn`/`redeem` under `scope='rm'`; referrer `referral*` distinct reasons; dispensary under `scope='dispensary:<pid>'`; synthetic `imgpick_*` / `review:*` refs) — so legitimately-distinct entries differ in at least one indexed column.

**Prod-duplicate prerequisite (blocking):** creating the UNIQUE index fails if prod `points_ledger` already contains duplicate `(order_ref, reason, scope)` rows. Before the index migration reaches prod, we MUST (a) query prod for existing duplicates, and (b) if any exist, dedup them (keep the earliest, delete the rest, adjust no balances — they are erroneous double-entries) via a one-off owner-gated operation. The migration must not silently swallow an index-creation failure (that would leave I2 open unnoticed) — post-deploy we verify the index exists in prod.

**Amount-of-work note on `_add` return contract:** callers of `credit`/`earn`/`redeem`/`spend` rely on the returned balance and on `balance_after` being written monotonically. With `INSERT OR IGNORE`, a losing concurrent insert must NOT recompute/rewrite `balance_after`. The implementation reads the post-insert state (whether it inserted or ignored) and returns consistently; `redeem` must not debit twice. This is the delicate part and gets the most test coverage.

## Out of scope (YAGNI)
- Persisting Stripe `session_id`/metadata on the order, and a settlement heal-sweep. The webhook backfill covers every realistic crash path; the only gap is Stripe never delivering the event at all (rare, and it would also block receipt booking), not worth the machinery now.
- The `wallet.redeem_for_module` latent-vs-dead question (separate follow-on).

## Testing
- **Marker:** `mark_order_settled` sets `settled_at` once (idempotent); order reads expose it.
- **I1 webhook backfill:** a paid-only order with the receipt already booked but `settled_at IS NULL` → webhook runs settlement (per kind) and marks settled; a `settled_at`-set order → webhook skips settlement; best-effort (settler raises → still 200, still booked).
- **I2 atomic points:** two `credit`/`earn` with the same `(order_ref, reason, scope)` yield ONE ledger row (the second is an atomic no-op, not a second insert), including a simulated race that bypasses the `has_entry` fast-path; `redeem` never debits twice; distinct `(order_ref, reason, scope)` still insert independently; the `imgpick_*`/`dispensary`/referral paths still record correctly.
- **Redirect unchanged:** existing per-kind settlement + characterization tests stay green; `settled_at` now set after.
- **Migration idempotency:** `init_orders_table`/`init_points_table` re-run safe; UNIQUE index creation is idempotent.

## Files
- **Modify:** `dashboard/orders.py` (settled_at column + `mark_order_settled` + expose on reads), `dashboard/points.py` (UNIQUE index + `INSERT OR IGNORE` + return contract), `app.py` (webhook decouple + mark settled; redirect mark settled).
- **Create:** `tests/test_settled_marker.py`, `tests/test_points_atomic.py`; extend `tests/test_webhook_back_booking.py`.
- **One-off (prereq):** prod points-ledger duplicate check + dedup (owner-gated), before the index migration deploys.
