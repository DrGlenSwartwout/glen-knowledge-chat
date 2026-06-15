# Patient channel-locked loyalty points

A patient earns and redeems RM loyalty points on a practitioner's client page
(`/dispensary/<code>`), scoped to that practitioner. Ships dark behind the
`CLIENT_POINTS_ENABLED` flag (default off).

## Earn
- On a paid client-page order, the patient earns `points_earn_pct` (5%, the same
  console-editable rate as retail) on the product subtotal.
- **Full-price only:** no earn on an order where the patient redeemed points (mirrors the
  retail earn rule).
- **Card payments only** (the Stripe checkout-return path that settles on paid, the same
  place the practitioner's margin is credited). Alt-pay (Zelle/Wise) dispensary orders are
  reconciled manually and do not earn/redeem in v1.
- Recorded idempotently per invoice (`reason="earn:dispensary"`, scope `dispensary:<pid>`).

## Scope (channel-locked)
- Points are stored on the shared loyalty ledger with a `scope` column. Ordinary retail
  points are `scope="rm"`; channel points are `scope="dispensary:<practitioner_id>"`.
- A patient's channel points are redeemable **only** on that practitioner's client page.
  They never apply to RM-direct retail/funnel checkout, and a patient who shops two
  practitioners has two separate balances (no cross-redemption).

## Redeem (RM-absorbed, fee-capped)
- At checkout the patient can apply their channel balance. The redemption is applied as a
  fixed `discount_cents` on the patient's invoice, so the patient pays less.
- **RM absorbs it; the practitioner keeps full margin.** RM collects the patient's payment,
  keeps `base + fee`, and credits the practitioner the full margin (`selling − base − fee`).
  A redemption comes out of RM's cut, never the practitioner's.
- **Safety cap:** redemption is capped at the order's **total service fee**
  (`min(requested, scoped_balance, total_fee_cents)`). This guarantees RM never sells a
  bottle below its blended base cost — RM forgoes at most its service margin.
- Recorded idempotently on paid (`reason="redeem"`, scope `dispensary:<pid>`); the settle
  takes `min(redeemed, current scoped balance)` to stay safe if the balance changed between
  checkout and payment.

## Endpoints / UI
- `POST /api/client/<code>/points` `{email}` (consent-gated) → `{ok, balance_cents,
  client_points_enabled}` for the patient's `dispensary:<pid>` scope.
- `/api/client/<code>/catalog` includes `client_points_enabled`.
- `/api/client/<code>/checkout` accepts `points_to_redeem_cents` in its body (capped server-side).
- The client page shows "You have $X in <practice> rewards" and an "Apply my points"
  control when the flag is on and the balance is positive.

## Flag
`CLIENT_POINTS_ENABLED` (env, default off) gates earn, redeem, the balance endpoint's
lookup, and the page UI. Off → no points behavior, no UI, zero change to the order flow.

## Deferred (v2)
- A more generous "subsidize below base" redemption option (a console toggle).
- Refund-driven reversal of earned/redeemed channel points.
- Earn/redeem on alt-pay (Zelle/Wise) orders.
- Cross-scope portability.
