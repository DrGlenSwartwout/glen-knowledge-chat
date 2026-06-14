# Rewards tiers + referral attribution + cash-out

Rewards referrers when a referred buyer purchases. Entirely behind `REWARDS_TIERS_ENABLED`
(default OFF — it moves real money).

## Tiers (read from `people.tags`)
- **Pro-influencer** — tag `tier:pro-influencer` → **cash** (accrued in `affiliate_earnings`, owner-approved payout).
- **Doctor / practitioner** (`type:practitioner`) and **client affiliate** (everyone else) → **points** credited to the referrer's own balance (spend on their own orders).
- Tier resolution: `dashboard/rewards.reward_mode_for_slug(cx, slug)` → `"cash"` if `tier:pro-influencer`, else `"points"`.

## Crediting (on a paid order, via `_settle_referral` in `/begin/checkout-return`)
- Referrer looked up from existing attribution (`referral_events` → `affiliate_signups` slug, by buyer email).
- Reward = `referral_reward_pct` × product spend (`total − shipping − GET`).
- Only **full-price referred** orders credit (`discount_cents == 0`); **self-referral excluded**.
- **Idempotent per order** (points `has_entry(order_ref,"referral")`; cash `UNIQUE(slug,order_ref)`).
- **Buyer first-order suppression (decision b):** an affiliate-acquired buyer earns NO points on their first order (the affiliate owns that acquisition); they earn normally afterward.

## Cash-out (review-gated, never automatic)
- When a referrer's pending cash (cash mode) or points balance (points mode) crosses
  `cash_out_threshold_cents`, a **review todo** is raised for Glen (idempotent per threshold band).
- Payout is the `rewards.process_payout` action at **`MONEY_SEND`** risk — routed through the
  dispatch policy to **owner approval** before anything is recorded. Points cash out at
  `cash_out_face_pct` (default 0.70 — points are worth more spent on product than cashed out).
  The action records intent (`mark_paid` / points redeem); the actual money-send is the owner's
  existing finance flow. Capture a **W-9** at this review step for 1099 tracking.

## Settings (DEFAULTS — confirm before enabling)
In `pricing-settings.json` (or the rewards settings store):
- `referral_reward_pct` = **0.05** (5% of the referred order's product spend)
- `cash_out_threshold_cents` = **10000** ($100)
- `cash_out_face_pct` = **0.70** (points→cash haircut)

## Known limitations (backlog / Plan 6)
- **No refund/charge-back reversal.** If a referred order is later refunded, the referrer keeps the
  credit. Cash mode is caught at the owner approval step (the cash-out is just a review prompt);
  **points-mode credit is currently permanent** — exposure is bounded by the small reward size.
  A `referral`-reversal entry on refund is the proper fix.
- **Points-mode payout idempotency** relies on the balance→0 effect after the first redeem (a second
  approved payout redeems 0) rather than a deterministic dedup key. Owner-gated, low risk.
- **Subscriptions don't generate referral rewards** (only one-time full-price referred orders).
- `tier:pro-influencer` is set manually (in the console/CRM on approval) and should be synced to GHL.

## Go-live
1. Confirm the three settings above.
2. Tag your professional influencers `tier:pro-influencer` (everyone else defaults to points).
3. Set `REWARDS_TIERS_ENABLED=true` in Doppler + Render.
4. Watch the first few cash-out review todos before approving any payout.
