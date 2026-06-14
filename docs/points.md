# Loyalty points — earning + redemption

The customer points loop. Ledger: `dashboard/points.py` (`points_ledger`, value in redemption-value
cents, 1 point = 5c). Pricing/redemption flows through the Plan 1 engine + Plan 2 checkout.

## Earning
- **5% of PRODUCT spend on a paid full-price order.** Settled at `/begin/checkout-return` (which
  already detects `payment_status == "paid"` and looks up the order) via `_settle_order_points`.
- "Full-price" = the order had **no discount and used no points** (`discount_cents == 0 AND
  points_redeemed_cents == 0`). A volume/coupon/points order earns nothing.
- Earn base = `total_cents − shipping_cents − get_cents` (shipping and absorbed GET excluded).
- **Idempotent** per invoice (`points.has_entry(order_ref, "earn")`) — re-hitting checkout-return
  never double-credits.
- Subscriptions never earn (they go through the scheduler, not checkout-return, and are discounted).
- Rate is `points_earn_pct` in `pricing-settings.json` (default 0.05).

## Redemption
- The reorder cart shows the balance (`GET /api/points/balance`) and an "apply points" control.
- `/reorder/checkout` caps the requested redemption at the caller's current balance, then the engine
  caps it again at the points floor (43% of list). The recorded `points_redeemed_cents` is what the
  engine ACTUALLY applied.
- The ledger is **deducted on confirmed payment** (the same `_settle_order_points` hook, guarded by
  `has_entry(order_ref, "redeem")`), not at checkout creation — so an abandoned checkout never spends
  points.

## Known limitations (Plan 5 / backlog)
- **Concurrent redemption window:** two checkouts fired within the settlement gap can both apply a
  redemption discount while the ledger only deducts once (the second `redeem` raises and is swallowed
  so the order isn't blocked). Exposure is bounded by the (small) balance. A reservation row or
  settle-time re-validation would close it.
- **Begin-funnel earning:** the funnel direct-buy path (`/begin/checkout`) isn't on the engine yet, so
  it doesn't record `discount_cents`; until it's wired to `_price_cart`, a discounted-via-tier funnel
  order would still earn. Single-bottle full-price funnel buys earn correctly.

## Deferred to Plan 5 (rewards tiers)
Affiliate-acquired first-order suppression (needs attribution); referral crediting (referrer earns
points if client-affiliate/doctor, cash-queue if pro influencer); tier from People-hub/GHL tags;
the points→cash cash-out (~70% of face) via a threshold-triggered payout-review task + W-9.
