# Continuous Care Redesign — Monthly Auto-Charge + Term Cap — Design

**Date:** 2026-07-02
**Part of:** the healing-first offer redesign (see `project_membership_prepay_ladder`).
**Decisions locked by Glen 2026-07-02:** monthly = auto-charge card on file; monthly rate = flat
$99/mo; up-front payment earns the discount; retire the 3-month tier (keep 1mo).

## Goal

Continuous Care becomes a **6- or 12-month commitment** with two ways to pay:

- **Monthly:** $99/mo **auto-charged** to a card on file, for exactly the committed term
  (6 or 12 charges), then it **stops** — no auto-renew. Total $594 (6mo) / $1,188 (12mo).
- **Up front (discounted):** 6mo **$546**, 12mo **$990** ("2 months free"). The discount is
  the reward for paying up front. This path already exists (prepay day-based grant).
- **1-month — $99** stays as a light single-term entry (no commitment, no discount).
- **3-month tier retired** ("for now" — kept trivially restorable).

## What already exists (reuse unchanged)

The group-bundle $99/mo membership proves the whole recurring path:
- `stripe_pay.create_checkout_session(..., save_card=True)` vaults the card (mode=payment +
  `customer_creation=always` + `setup_future_usage=off_session`).
- `subscriptions.create_membership(...)` inserts the `kind='membership'` row.
- `cron_charge_subscriptions` (app.py:22750) membership branch: off-session `charge_off_session`
  on the stored customer+payment_method, `advance_after_charge` (order_count++, next_charge +=
  cadence), extends the `memberships` grant (`_extend_membership_grant`, keeps portal access in
  sync), dunning (3 fails → `past_due` + email), and the FTC cancel-link heads-up pass.
- Idempotent fulfillment: claim-then-create on a `*_grants(session_id PK)` table + PaymentIntent
  re-fetch, called from both `/…/return` AND `/webhook/stripe` (closed-tab safety net).
- Both a day-based grant (up-front) and a subscriptions row (monthly) read as a paid member —
  `_is_paid_member` converges correctly for both.

## The one real gap: a TERM CAP

Nothing today limits how many times a `kind='membership'` sub is charged — the cron bills it
forever until manually cancelled or 3 failures. Continuous Care monthly MUST stop after 6/12
charges. This is the load-bearing new piece.

## Components

### 1. `subscriptions` schema + `create_membership` (dashboard/subscriptions.py)
- New idempotent migration `migrate_add_term_cap_column(cx)` → `term_charges_total INTEGER`
  (nullable; NULL = uncapped, so every existing group-bundle sub is unaffected).
- `create_membership(...)` gains two backward-compatible kwargs:
  `term_charges_total=None` and `initial_order_count=0` (written into the row; existing callers
  keep NULL cap + order_count 0).

### 2. Charge-cron cap enforcement (app.py `cron_charge_subscriptions`, membership branch)
Immediately **after** the successful `advance_after_charge` + grant-extend, using the already
re-read `updated` row: if `updated["term_charges_total"]` is set and
`updated["order_count"] >= updated["term_charges_total"]` → `set_status(cx, sid, "cancelled")`
(the existing manual-cancel path; it zeroes order_count and stamps cancelled_at, fine since the
term is over). This runs on the SAME success branch that sends the receipt, so a "final payment —
your term is complete" note can be added there (nice-to-have; a plain receipt is acceptable v1).

### 3. Monthly checkout + fulfiller (app.py), flag `CONTINUOUS_CARE_MONTHLY_ENABLED`
- `POST /continuous-care/checkout` — validate `term_months ∈ {6, 12}`; flag off or not
  `_STRIPE_ACTIVE` → `{ok:False}`/200 (mirror `prepay_checkout`). Build
  `create_checkout_session(MONTHLY_ANCHOR_CENTS, save_card=True,
  metadata={"kind":"continuous_care_monthly","email","term_months"},
  success_url=".../continuous-care/return?session_id={CHECKOUT_SESSION_ID}")`.
  **Charges month 1 ($99) now** and vaults the card.
- `GET /continuous-care/return` → `_fulfill_continuous_care_monthly(session_id)`.
- `_fulfill_continuous_care_monthly(session_id)` — mirror `_fulfill_prepay_term`'s security +
  idempotency: re-fetch session + PI, require `pi.status == "succeeded"`, claim-then-create on
  `continuous_care_grants(session_id TEXT PRIMARY KEY)`. Pull `customer`/`payment_method` off the
  PI. Then:
  - `create_membership(cx, email, stripe_customer_id=customer, stripe_payment_method_id=pm,
    amount_cents=MONTHLY_ANCHOR_CENTS, next_charge_date=add_months(today, 1), cadence_months=1,
    term_charges_total=term_months, initial_order_count=1)` — order_count=1 records the month-1
    charge just taken (so `category_for` reads `full`, not `trial`), and the cap counts TOTAL
    months (checkout month 1 + cron months 2..N = N charges; cron cancels when order_count
    reaches N).
  - `_grant_membership(cx, email, ~35 days, "continuous_care")` for immediate portal access until
    the first cron charge extends it.
  - Mint the FTC one-click cancel token (`auth_tokens`, purpose `membership_cancel`), same as the
    removed biofield-trial block.
  - Best-effort `_ingest_order(source="continuous_care_monthly", ...)` + confirmation email.
- Wire `_fulfill_continuous_care_monthly(session_id)` into `/webhook/stripe` alongside the other
  fulfillers (each no-ops on non-matching kind).

### 4. prepay.py — retire 3mo, expose monthly option data
- `PUBLIC_TIER_KEYS = ["1mo", "6mo", "12mo"]`; `tiers_public()` filters to these (3mo stays in
  `TIERS` for trivial restore, just not offered). 
- Add helpers so the picker can show both options for 6/12: `monthly_total_cents(key)` (= 9900 ×
  months) and `upfront_savings_cents(key)` (= monthly_total − price_cents). `tiers_public()` adds
  `monthly_cents` (9900), `monthly_total_cents`, `upfront_savings_cents`, and a `commitment`
  bool (True for 6/12) per tier.

### 5. Picker UI (static/prepay.html)
- 1mo → single "Continue" (existing up-front path).
- 6mo/12mo → two actions: **Pay monthly ($99/mo)** → `POST /continuous-care/checkout
  {term_months}`; **Pay up front ($546 / $990, save $X)** → existing `POST /prepay/checkout
  {tier_key}`. The monthly buttons render only when `CONTINUOUS_CARE_MONTHLY_ENABLED` (surfaced
  via a small injected flag or the tiers payload); off → up-front only (today's behavior).

## What this does NOT change
- Up-front prepay path, the $1 deposit (Model #2 stays no-auto-charge), pricing engine, the
  two-door surface. Existing group-bundle membership (uncapped) is untouched (NULL cap).

## Deploy-safety
- `CONTINUOUS_CARE_MONTHLY_ENABLED` default off → no monthly checkout, picker shows up-front only.
- `term_charges_total` NULL for every existing sub → cron behavior for current subs byte-identical.
- The migration is idempotent/additive.

## Testing (money-critical — high coverage)
- **prepay model:** `tiers_public()` excludes 3mo, includes 1/6/12 with monthly_total +
  upfront_savings; `monthly_total_cents`/`upfront_savings_cents` correct for 6/12.
- **term cap unit (subscriptions.py):** `create_membership(term_charges_total=6,
  initial_order_count=1)` writes both; migration idempotent.
- **cron cap:** a capped membership sub charged repeatedly cancels exactly when order_count
  reaches the cap — charges N-1 cron times (months 2..N) then `status='cancelled'`, never an
  N+1th charge. An uncapped (NULL) sub still charges indefinitely (regression guard).
- **checkout:** flag off → `{ok:False}`; flag on → session at 9900 with `save_card=True` +
  correct metadata (monkeypatch `create_checkout_session` to capture).
- **fulfiller:** `/continuous-care/return` creates a membership sub with the right cap +
  order_count=1 + stored customer/pm, grants access (`_is_paid_member` True, category `full`),
  idempotent on replay (no double sub, `continuous_care_grants` claim), mints a cancel token,
  writes no sub when PI not succeeded.

## Follow-on (NOT this build)
- Member-wide reorder discount ("discounts on all reorders, not just test-recommended") — a
  discount-controls (#506) gating change; separate PR.
- Renewal-prompt branch for `continuous_care` source in `cron_membership_renewals` when a term
  completes (nicety).
