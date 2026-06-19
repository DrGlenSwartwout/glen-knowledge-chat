# Spec 2b-2 — Referrer reward (double-sided, % of referee's order as points)

**Date:** 2026-06-19
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`)
**Parent:** Spec 2b. Fast-follow to 2b-1 (merged, PR #184). Completes the double-sided referral loop.

---

## Problem

2b-1 gives the referee 10% off. 2b-2 closes the loop: when a referee's referral order is **paid**, the referrer earns store-credit points equal to a configurable **percent of the referee's product spend** on that order. Conversion-based, idempotent, once per distinct referee, referrers uncapped.

## Scope (2b-2)

A reward step inside the existing paid-order hook (`_settle_order_points`): on payment, find the order's referral redemption, credit the referrer points = `REFERRER_REWARD_PCT% × the referee's product spend`, and stamp the redemption as rewarded. No checkout/pricing change.

**Out of scope:** per-SKU / link-scoped affiliate rewards (whole initial order only — a parked future idea); refund clawback of an already-paid reward; a referrer-facing dashboard of earnings (the data is recorded for a future view).

---

## Confirmed decisions (Glen, 2026-06-19)

- **Reward = store-credit points** (reuses the points ledger; points already beat cash).
- **By percent, not fixed $:** referrer earns **`REFERRER_REWARD_PCT`% of the referee's product spend** on the referral order (= `total_cents - shipping_cents - get_cents`, the same "product_cents" basis the existing points-earn uses — excludes shipping/tax/GET).
- **Base = the whole initial (referral) order** for now; per-SKU scoping parked.
- **Config doubles as the on-switch:** `REFERRER_REWARD_PCT` (default **0 = off**). Inert until set; rides under `REFERRALS`.
- **Idempotent, once per referee, referrers uncapped.**

---

## Architecture

### Store — extend `dashboard/referrals.py`
- Additive columns on `referral_redemptions` (lazy `ALTER`): `rewarded_at TEXT DEFAULT ''`, `reward_cents INTEGER DEFAULT 0`.
- New functions:
  - `redemption_by_order_ref(cx, order_ref) -> dict|None` — the redemption row (referee_email, owner_email, code, rewarded_at, ...) whose `order_ref` matches; None if none.
  - `mark_rewarded(cx, referee_email, reward_cents) -> None` — stamps `rewarded_at=now`, `reward_cents`.

### Reward helper + config (app.py)
- `_referrer_reward_pct() -> int` — reads env `REFERRER_REWARD_PCT` (default 0; clamps ≥0).
- `_settle_referrer_reward(cx, order, order_ref) -> int` — the reward step (returns cents credited, 0 if none):
  1. Flag/config gate: if not `_REFERRALS` or `_referrer_reward_pct() <= 0` → return 0.
  2. `red = referrals.redemption_by_order_ref(cx, order_ref)`; if none or `red["rewarded_at"]` already set or `red["owner_email"]` falsy → return 0.
  3. Compute `product_cents = max(0, total_cents - shipping_cents - get_cents)` from `order`; `reward = product_cents * pct // 100`; if `reward <= 0` → still mark rewarded (so it doesn't retry) and return 0.
  4. `points.credit(cx, red["owner_email"], value_cents=reward, reason="referral_reward", order_ref=f"referral:{red['referee_email']}")` (idempotent: one reward per distinct referee, ever).
  5. `referrals.mark_rewarded(cx, red["referee_email"], reward)`; return `reward`.
- Called from `_settle_order_points` inside its existing `with sqlite3.connect(LOG_DB) as cx:` block (after the buyer earn/redeem logic), wrapped in its own try/except so a referrer-reward failure never affects the buyer's settle or the payment return.

### Idempotency (two layers)
- `points.credit` keyed on `order_ref=f"referral:{referee_email}"` + `reason="referral_reward"` → the same referee can never trigger a second referrer credit (even across re-settles or a second order).
- The `rewarded_at` stamp short-circuits the lookup on re-entry. `_settle_order_points` itself is already idempotent per order_ref.

### Flag/config
No new boolean flag. Gated by `REFERRALS` (the redemption only exists when 2b-1 is on) AND `REFERRER_REWARD_PCT > 0`. With the pct at 0 (default), 2b-2 is fully inert — no credit, no stamp.

---

## Data flow
1. A referee checks out with a code (2b-1) → a `referral_redemptions` row with `order_ref` + `owner_email`.
2. That order is **paid** → `_settle_order_points(order, order_ref)` runs → `_settle_referrer_reward` finds the redemption, credits the referrer `pct% × product_cents`, stamps `rewarded_at`.
3. The referrer's points balance reflects the reward; it's recorded on the redemption (`reward_cents`) for a future earnings view.

## Error handling
- No redemption for this order_ref, already rewarded, missing owner, pct 0, or product_cents 0 → no credit (and on a 0-value computation, still stamp to avoid pointless retries).
- The reward step is wrapped so an exception is logged and never blocks the buyer's points settle or the payment-return handler.
- Idempotent credit + `rewarded_at` make re-settles / duplicate payment webhooks safe.
- Refund of the referee's order after the reward is paid → NOT clawed back in 2b-2 (noted; future).

## Testing
- **Store:** additive columns present; `redemption_by_order_ref` returns the row for a matching order_ref, None otherwise; `mark_rewarded` stamps `rewarded_at` + `reward_cents`.
- **Reward helper (`_settle_referrer_reward`, direct unit test with a seeded redemption + order dict):**
  - valid: credits the referrer `pct% × product_cents`, stamps rewarded, returns the cents; the referrer's `points.balance` reflects it.
  - idempotent: a second call (same referee/order) credits nothing more (balance unchanged).
  - no redemption for the order_ref → 0, no credit.
  - `REFERRER_REWARD_PCT=0` → 0, no credit, no stamp.
  - product_cents 0 (e.g. all shipping) → 0 credit, but stamped (no retry).
- **Settle integration:** `_settle_order_points` on a paid referral order credits the referrer once (asserts balance) and does not affect the buyer's own earn/redeem logic; a non-referral order is unaffected.
- Follow deploy-chat test isolation (tmp `$DATA_DIR`; mock Supabase; importorskip playwright; `importlib.reload`). NO emoji; no em dashes.

## Flags / config
`REFERRALS` (existing) + `REFERRER_REWARD_PCT` (default 0 = off; set e.g. `10`). Both required for any reward. `REFERRER_REWARD_PCT=0` → fully inert.

## Notes
- Reuses the 2b-1 `referral_redemptions` table (additive columns only), `points.credit` (idempotent), and the existing `_settle_order_points` paid-order hook. No checkout/pricing change, no new external dependency.
- The reward base nets out shipping + GET (matching the buyer points-earn rule), so the referrer earns only on the referee's product spend.
- `reward_cents` + `rewarded_at` on each redemption are the seed for a future referrer earnings dashboard (out of scope here).
