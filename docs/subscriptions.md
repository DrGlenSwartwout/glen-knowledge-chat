# Subscriptions — "Subscribe & Grow"

Recurring orders where Stripe vaults the card and OUR daily scheduler charges off-session
each cycle through the Plan 2 pricing path (`_price_cart`) at an escalating loyalty tier.

## Flags (all must be on to charge real cards)
- `SUBSCRIPTIONS_ENABLED` — turns the whole system on (setup endpoint, scheduler, portal). Default OFF.
- `PRICING_ENGINE_CHECKOUT` — subscriptions price through `_price_cart`, so this must be on too.
- `STRIPE_ACTIVE` + `STRIPE_SECRET_KEY` — real Stripe (vault + off-session charge).

## Model
- **Vault, not Stripe Subscriptions:** the first order is a normal Stripe Checkout with
  `save_card=True` (`setup_future_usage=off_session` + a Stripe Customer). On checkout-return
  we read the Customer + payment-method off the PaymentIntent and write a `subscriptions` row.
- **Daily scheduler** (`POST /api/cron/charge-subscriptions`, `X-Cron-Secret` gated): a heads-up
  pass (email ~3 days before a charge) then a charge pass. Each due sub is priced via
  `_price_cart(subscriber_tier_pct=tier_for(order_count))`, charged **subtotal + shipping**
  off-session (GET absorbed, not billed), then order + QBO invoice + receipt; tier advances.
  Wired into the `glen-pb-tag-sync-daily` cron script (no new Render cron service, to respect
  the Blueprint cron limit) + a standalone `scripts/run_subscriptions_cron.py` for manual runs.
- **Escalating tier (by completed-order count):** order #1 (the sign-up/setup order) = 5%,
  order #2 = 10%, order #3+ = 15%. The setup order counts as order #1, so the subscription is
  created with `order_count=1` and the first SCHEDULED charge is order #2 at 10%. Skip + pause
  HOLD the tier; cancel resets it.
- **Skip/pause/cadence/cancel:** self-serve at `/subscription` (magic-link via the reorder cookie).
  Card updates show a "contact us" note in v1.
- **Dunning:** a failed/requires-action charge bumps `failed_count`, emails the customer, and never
  advances; at 3 fails the sub goes `past_due` (stops being charged).

## Safety
- Whole system is flag-gated (default off). The scheduler supports `?dry_run=1` (computes + logs,
  charges/mutates nothing). Always dry-run first after enabling.
- Never advances a subscription on a failed charge. Due list is snapshotted before the skip pass,
  so a consumed skip can never be billed in the same run. Each sub is processed in its own
  try/except. The setup endpoint guards on `_STRIPE_ACTIVE` before creating an invoice.

## Go-live checklist
1. Confirm `STRIPE_SECRET_KEY` (live) in Doppler `remedy-match/prd` + Render.
2. Set `PRICING_ENGINE_CHECKOUT=true`, `STRIPE_ACTIVE=true`, then `SUBSCRIPTIONS_ENABLED=true`.
3. Run the scheduler once with `?dry_run=1` and read the log summary before the first real charge.
4. Test one real subscription end-to-end (subscribe → next-cycle charge → skip → cancel).

## Known v1 limitations (later plans)
Full SCA/3DS off-session confirmation (v1 treats `requires_action` as a failure + notify);
in-portal card update (v1 = contact us); auto-points-redeem on subscription orders; the
rewards-tier referral attribution + cash-out (Plan 4). The `glen-pb-tag-sync-daily` cron now
also runs billing — rename it to something neutral (e.g. `glen-daily-ops-cron`) next time
render.yaml is touched, so the dashboard label isn't misleading.
