# Shipping-credit: auto-apply to next order + one-click refund

**Date:** 2026-07-12
**Status:** Approved shape + scope (2026-07-12); build behind a dark flag
**Predecessor:** `2026-07-12-recalc-shipping-overpay-credit-design.md` (#831, merged) —
that slice records a paid combined-shipment member's shipping overpayment in the
per-order `overpay_credit_cents` column (informational only). This slice makes that
credit actionable.

## Goal

An already-paid customer who overpaid on shipping should have that credit
**auto-apply to their next order** (default), with a console **one-click refund**
as the override. Auto-apply fires on **every** next-order flow: in-house order
entry, portal reorder, and the public funnel checkout (Glen's choice: "everywhere
they buy next").

## Why not the existing points path
`points_ledger` gives an email-keyed, idempotent, spendable balance — the right
store. But the loyalty-points *redemption* path floors every line at 43% of list
(`pricing.py` `points_floor_pct`), and is opt-in per checkout. A shipping refund
must apply **in full** and **automatically**. So we reuse `points_ledger` only as
the balance store (a dedicated `ship_credit` scope, isolated from `rm` loyalty
points) and apply the credit as its own uncapped credit line, not through
`apply_points`.

## Feature flag
`SHIP_CREDIT_AUTOAPPLY_ENABLED` (Doppler prd, default OFF). Gates grant + auto-apply
+ refund UI. Ships fully dark; enable atomically after per-flow verification. When
off, every new code path is a no-op and behavior is exactly slice-1.

## Components

### 1. `dashboard/ship_credit.py` — the credit ledger wrapper (pure-ish, testable)
Thin wrapper over `dashboard/points.py` primitives, scope `"ship_credit"`:
- `grant(cx, email, cents, *, source_ref)` → `points.credit(... reason="ship_overpay",
  order_ref=source_ref, scope="ship_credit")`. Idempotent on (source_ref, reason,
  scope) via points' `has_entry` guard — re-running recalc never double-grants.
- `balance(cx, email)` → `points.balance(cx, email, scope="ship_credit")`.
- `plan_application(balance_cents, chargeable_cents)` → **pure**:
  `max(0, min(balance_cents, chargeable_cents))` — never exceeds the balance or the
  order's own chargeable total (can't make a total negative).
- `consume(cx, email, cents, *, applied_ref)` → guarded debit: no-op if an entry with
  (applied_ref, reason="ship_credit_applied", scope) already exists (idempotent on the
  APPLYING order, so re-pricing/resubmitting one order never double-spends); else
  `points.redeem(... order_ref=applied_ref)` clamped to current balance.
- `mark_refunded(cx, email, cents, *, source_ref)` → guarded debit with
  reason="ship_credit_refunded" (removes the balance so a refunded credit can't also
  auto-apply; the guard is the already-refunded check).

### 2. Grant — at recalc (`app._recompute_combined_shipping`)
After computing `credit` for a paid member (slice-1 logic, unchanged), if
`SHIP_CREDIT_AUTOAPPLY_ENABLED` and `credit > 0` and the member has an email:
`ship_credit.grant(cx, email, credit, source_ref=<paid order external_ref>)`.
`overpay_credit_cents` stays as the per-order audit/display value.

### 3. Auto-apply — at each checkout (flag-gated)
New order column `ship_credit_applied_cents INTEGER NOT NULL DEFAULT 0`. In each
order-creation flow, before computing the charged amount:
```
apply, note = _apply_ship_credit(cx, email, chargeable_cents)  # returns 0 when flag OFF
```
where `_apply_ship_credit` checks `SHIP_CREDIT_AUTOAPPLY_ENABLED` first (returns
`(0, None)` when off), else `plan_application(ship_credit.balance(cx, email),
chargeable_cents)`. `plan_application` itself is pure and flag-agnostic.
Subtract `apply` from the amount charged + stored total, store it in
`ship_credit_applied_cents`, and render an invoice line "Shipping credit applied:
−$X". On order persist/payment, `ship_credit.consume(cx, email, apply,
applied_ref=<new order external_ref>)` (guarded idempotent). Flows to touch (each
already has a points-redeem hook to mirror — see the `PROGRAM_CARE_TASTER`
auto-default precedent at `app.py:7813`):
- funnel `/begin/checkout/<slug>` (`app.py:~7890`)
- reorder `_checkout_cart` (`app.py:~23085`)
- portal/client `/api/client/<code>/checkout` (`app.py:~14628`)
- in-house `_price_inhouse_invoice` (`app.py:~35423`)
- practitioner checkouts share `_price_cart`/`_checkout_cart`.

A single shared helper `_apply_ship_credit(cx, email, chargeable_cents)` returning
`(apply_cents, note)` keeps each call site to ~2 lines.

### 4. One-click refund (flag-gated)
New governed action `finance.refund_ship_credit` (module "money", MONEY_SEND,
confirm-gated, OWNER/OPS execute · VA queues), wrapping slice-1 refund logic:
- amount = the order's outstanding shipping credit (its `overpay_credit_cents`, minus
  anything already applied/refunded — resolved via the ledger balance for that email
  bounded by this order's grant).
- calls `_refund_order_exec` (Stripe-then-QBO; non-card → QBO money-out + manual send).
- `ship_credit.mark_refunded(...)` zeroes the balance so it won't auto-apply.
- **already-refunded guard** (the `mark_refunded` guard) — fixes the existing gap
  where `finance.refund_order` has no double-refund protection.

Surfaced as a "Refund credit" button on `console-orders.html` order cards that carry
`overpay_credit_cents > 0` and are not yet refunded (flag-gated).

## Idempotency & edge cases
- Grant idempotent on source order; consume idempotent on applying order; refund
  guarded on source order. No path double-counts.
- `plan_application` clamps so a credit never exceeds the new order's total (no
  negative charge). Remainder stays as balance for the following order.
- Non-card original payment → refund books a QBO money-out receipt; Rae settles
  manually (the confirm dialog says so). Card → Stripe auto-refund.
- Flag OFF → grant/apply/refund all no-op; identical to slice-1.

## Testing
- `tests/test_ship_credit.py`: grant idempotent; balance; `plan_application`
  clamps to balance and to chargeable and floors at 0; consume guarded-idempotent +
  clamps; mark_refunded guarded.
- Integration (doppler dev, temp DB): in-house flow applies + consumes a credit and
  reduces the charge; second submit of the same order doesn't double-consume; refund
  action zeroes the balance and blocks a second refund. At least one card-charging
  flow (reorder) exercised for the applied-line + consume.
- Flag-off regression: every new path is a no-op; slice-1 tests still green.

## Rollout
Ship dark. Enable `SHIP_CREDIT_AUTOAPPLY_ENABLED` in prd after verifying each flow
on the live surface. Reversible (flip flag off).
