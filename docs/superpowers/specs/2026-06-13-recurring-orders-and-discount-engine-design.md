# Recurring Orders + Discount/Points Stacking Engine — Design Spec

**Date:** 2026-06-13
**Status:** Draft for Glen's review
**Builds on:** the reorder cart (PR #97 — `/reorder`, magic-link auth, `_get_product`,
`_qty_unit_cents`, `qb.create_invoice`, `_ingest_order`, `_tax.compute_get_cents`,
`/begin/checkout-return`).

Two features that must share one pricing core:
- **A. Discount & Points Stacking Engine** — the single source of truth for what a
  line/order actually costs after every discount, point, and floor.
- **B. Recurring Orders ("Subscribe & Grow")** — vault-the-card + our-own-scheduler
  subscriptions with an escalating loyalty discount.

They are specified together because B's per-cycle charge must price through A.

---

## Decisions locked with Glen (2026-06-13)

| Decision | Choice |
|---|---|
| Billing mechanism | **Stripe as card vault + our own daily scheduler** (off-session charges), NOT native Stripe Subscriptions |
| Cadences | Monthly / every 2 months / every 3 months |
| Loyalty discount | Escalating by completed-order count: **5% (1st) → 10% (2nd) → 15% (3rd+)** |
| Tier on skip/pause/cancel | Skip + pause **hold** the tier; cancel **resets** order_count to 0 |
| % discount stacking | **One % discount applies** — subscriber tier is **exclusive of coupons** (subscriber tier replaces coupons; never two % sources) |
| Points floor | Points may go **below wholesale** to **43% of list** ($30 on a $70 item) |
| Discount floor | All % discounts clamp at **57% of list** ($40 = wholesale on a $70 item) |
| Points earning | Earn on **full-price spend only** (not on discounted/subscription orders); redeem above the floor |
| Floor representation | **Percent-of-list** (57% / 43%), per-SKU absolute wholesale override allowed; all console-settable |
| Tax treatment | Points = a **price discount** (reduces the GET tax base), not a tender |

---

## Part A — Discount & Points Stacking Engine

### A.1 The principle
Floors are the **safety net**, not the primary control. The primary control is **which
discounts may combine**. Most orders should *not* land on a floor.

### A.2 Discount buckets and additivity
Three buckets. **At most one % discount total** (per the locked decision), then points.

| Bucket | Examples | Rule |
|---|---|---|
| Structural (earned) | Subscribe-&-save tier, member price | one applies |
| Promotional (public) | Daily coupon (`coupons.json`), seasonal sale, welcome | one, best-wins |
| Points | Loyalty redemption ($ off) | applies after the % discount |

**Exclusivity (locked):** Structural and Promotional do **not** stack. The order's single
percentage discount = the applicable structural discount **if any** (a subscription order
always uses its subscriber tier), otherwise the best promotional discount. Points then
apply on top of whatever % discount won.

- Subscription order → % = subscriber tier (5/10/15). Coupons ignored.
- One-time order, member → % = member discount.
- One-time order, non-member → % = best eligible coupon.
- (Open default, see §F: member-price vs coupon also follows best-one-wins.)

### A.3 Floors (percent of list, console-settable)
- `discount_floor_pct = 0.57` → price after % discount clamps **up** to `list × 0.57`
  (= wholesale; $40 on $70).
- `points_floor_pct = 0.43` → price after points clamps **up** to `list × 0.43`
  ($30 on $70).
- Per-SKU override: optional absolute `wholesale_cents` per product; when present it
  overrides `list × discount_floor_pct`, and the points floor becomes
  `wholesale_cents − points_allowance_cents` (default allowance = the gap implied by the
  two pct, i.e. `list × (0.57 − 0.43)`).

### A.4 Order of operations (also makes GET tax correct)
```
1. start: list_price (per line, via _get_product + _qty_unit_cents qty tiers)
2. apply the single winning % discount        → p1
3. clamp p1 up to discount_floor (57% / wholesale)   → p2
4. subtract points ($ allocated to this line) → p3
5. clamp p3 up to points_floor (43%)          → p4   (final unit/line price)
6. GET tax on Σ p4 via _tax.compute_get_cents (points already netted out of base)
7. shipping
```
Percentage discounts **compound** if two ever apply in a future policy change (honest +
margin-safe); today only one applies so it is moot.

### A.5 Points economics
- Earn: `points_earn_pct = 0.05` of **full-price** spend (console-settable). Discounted
  and subscription orders earn **nothing** (prevents compounding the giveaway).
- Redeem: `points_redeem_value` (default $1 per 20 points) as $ off, bounded by the
  points floor. The floor caps liability at `wholesale − points_floor` per item.
- Ledger is authoritative (earn/redeem rows), balance = running sum.

### A.6 The single pricing function (critical)
`pricing.compute(items, *, email, channel, subscriber_tier=None, coupon=None,
points_to_redeem=0) -> PriceResult` returning per-line breakdown, order subtotal,
discount_cents, points_redeemed_cents, get_cents, total_cents, and a human-readable
`breakdown` for the cart UI + invoice. **Used by all three callers** — cart preview,
one-time/reorder checkout, and the subscription scheduler — so they can never diverge.

---

## Part B — Recurring Orders ("Subscribe & Grow")

### B.1 Customer flow
1. In the reorder cart, after items + quantities, choose **One-time** or **Recurring**
   (+ cadence: monthly / 2-mo / 3-mo).
2. Inline incentive: *"Subscribe & grow your savings — 5% off today, rising to 15%. Skip,
   pause, or cancel anytime."* (doubles as the consent line).
3. **Consent checkbox:** "I authorize recurring charges and understand I can skip, pause,
   or cancel anytime." (tied to the ToS gate).
4. Checkout: Stripe Checkout `mode=payment` with `setup_future_usage=off_session` +
   customer creation → charges the first order (priced through A at tier 5%) **and** vaults
   the card.
5. `/begin/checkout-return` (extended) creates the `subscriptions` row.
6. **Heads-up email** `lead_days` (default 3) before each future charge: "ships in 3 days —
   skip, change, or pause."
7. Each cycle the scheduler charges off-session, creates the order/invoice, ships, emails a
   receipt, and advances the tier.

### B.2 Billing — vault + scheduler
- Stripe holds Customer + PaymentMethod (PCI stays on Stripe).
- Daily cron (`run_subscriptions_cron.py`, same launchd/in-process pattern as existing
  crons; consolidate per the render.yaml cron-limit note in memory):
  - **Heads-up pass:** active subs with `next_charge_date` within `lead_days` and not yet
    notified for this cycle → send heads-up.
  - **Charge pass:** active subs with `next_charge_date <= today` and `skip_next=false`:
    - price via A (subscriber tier, no coupon, no auto-points in v1);
    - off-session `PaymentIntent(off_session=true)`;
    - on success → `_ingest_order(source="subscription", subscription_id, channel)` +
      `qb.create_invoice` (discount line via `_discount_line`) + advance
      `next_charge_date += cadence` + `order_count += 1` (tier recompute) + receipt email
      + fulfillment (EasyPost/Rae);
    - `skip_next=true` → advance date, clear flag, **no order, no tier change**;
    - on failure → dunning (§B.5).

### B.3 Data model (chat_log.db)
- `subscriptions(id, email, stripe_customer_id, stripe_payment_method_id, items_json,
  cadence_months, status[active|paused|cancelled|past_due], order_count, next_charge_date,
  ship_address_json, skip_next, last_notified_cycle, created_at, updated_at, cancelled_at)`
- `points_ledger(id, email, delta, reason, order_ref, balance_after, created_at)`
- `pricing_settings` (or extend the existing JSON-config pattern): `discount_floor_pct`,
  `points_floor_pct`, `points_earn_pct`, `points_redeem_per_point_cents`,
  `points_allowance_cents`, `subscribe_tiers`, `cadences`, `lead_days`, `dunning_schedule`.
- Orders gain: `discount_cents`, `points_redeemed_cents`, `subscription_id`, `tier`.

### B.4 Manage-plan portal (hybrid)
Extends `static/reorder.html` (magic-link authed). Shows: next charge date, current + next
discount tier, items + quantities, cadence, points balance. Buttons: **Skip next**,
**Pause/Resume**, **Change cadence**, **Cancel**. Card/payment updates → **Stripe Customer
Portal** link (configured for payment-method management).

### B.5 Dunning (failed off-session charge)
Retry schedule `dunning_schedule` (default +1d, +3d, +5d). Each failure → "update your
card" email with the Stripe portal link. After the last retry → `status=past_due` (effectively
paused) + internal notify. `requires_action` (SCA/3DS) → email a one-tap confirm link rather
than hard-failing.

### B.6 Emails
Setup confirmation, pre-charge heads-up, per-cycle receipt, payment-failed/update-card,
cancel confirmation. Sent via the existing mailer (note: it is Gmail SMTP today; if these
should be GHL-tracked, route via GHL — flagged in §F).

---

## C. Edge cases & risks
- **SCA/3DS off-session** → email confirm link (handled in dunning).
- **Price / availability drift** → recompute at current price each cycle; an unavailable item
  is skipped with a notify, or the sub is paused + customer emailed (never silently dropped).
- **Address change** → portal edits `ship_address_json`; re-prices GET tax next cycle.
- **Card expiry** → heads-up + Stripe portal.
- **Refund / cancel mid-cycle** → existing Stripe refund path.
- **Wholesale channel** → points-below-wholesale is a deliberate, console-tunable choice;
  it is a private earned redemption, not a published price.
- **Floor sanity** → before launch, verify the 43% floor still clears variable cost
  (COGS + shipping + fees) on the heaviest / most-expensive SKUs, not just the $70 flagship.

## D. Console settings (all resettable)
`discount_floor_pct` (0.57), `points_floor_pct` (0.43), `points_earn_pct` (0.05),
`points_redeem_per_point_cents`, `points_allowance_cents`, per-SKU `wholesale_cents`
override, `subscribe_tiers` ([5,10,15]), `cadences` ([1,2,3]), `lead_days` (3),
`dunning_schedule` ([1,3,5]).

## E. Testing
- **Pricing engine** (the priority): table-driven tests over every combination — subscriber
  tier × coupon-exclusivity × points × both floors × qty tiers × GET tax base; assert the
  floors clamp and that points net out of the tax base.
- **Scheduler:** due / skip / pause / cadence-advance / tier-advance / dunning retries;
  Stripe + QBO stubbed (monkeypatch), LOG_DB to tmp.
- **Points ledger:** earn-on-full-price-only, redeem-bounded-by-floor, balance integrity.
- **Subscription lifecycle:** setup → charge → skip → cancel(reset tier).
- **Portal auth:** magic-link gating on all manage endpoints.
- Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat"
  ~/.venvs/deploy-chat311/bin/python -m pytest` (ignore the 2 known pre-existing failures).

## F. Open items (confirm or accept defaults)
1. **Member-price vs coupon** for non-subscribers — default best-one-wins (no stacking),
   consistent with the locked exclusivity. Confirm.
2. **Subscription receipt/heads-up emails** — send via current Gmail-SMTP mailer, or route
   through GHL so they are tracked in Conversations? Default: current mailer for v1.
3. **Auto-redeem points on subscription orders** — v1 = off (manual redemption only).

## G. Scope
- **v1:** fixed cart at signup; 3 cadences; 5/10/15 escalation; vault + scheduler off-session
  charge; pricing engine + floors + points ledger; heads-up + receipts + dunning; manage
  portal (skip/pause/cadence/cancel) + Stripe card portal; consent checkbox.
- **v2:** edit items mid-subscription; annual prepay; biofield-analysis-driven product swaps;
  auto-points-redeem on subscription; gift / multi-address.
