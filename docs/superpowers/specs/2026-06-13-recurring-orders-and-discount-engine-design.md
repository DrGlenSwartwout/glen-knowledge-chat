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
| % discount stacking | **One % discount applies** = the deepest of {**volume**, subscriber tier, coupon} (best-of-one). Subscriber tier is exclusive of coupons; volume is best-of with both. Never two % sources |
| Volume pricing | **Smooth months-based curve** (replaces 3/6/12 sharp tiers), **mix-and-match** across SKUs (total bottle-equiv "months" set one discount level), **format-decoupled** (price = months only) |
| Volume curve shape | Continuous, interpolated through anchors **1mo=100%, 3=86%, 6=71%, 12=57%(=wholesale), flat beyond**; front-loaded; console-tunable knots + cap; **flat-at-total-level** |
| Volume eligibility | **All Functional Formulations** (capsules + powders + liquids) on the same curve; capsules get format choice (bottle/larger/refill), powders+liquids individual bottles only. **Pure Powders excluded** (flat $40) |
| Points floor | Points may go **below wholesale** to **43% of list** ($30 on a $70 item) |
| Discount floor | All % discounts clamp at **57% of list** ($40 = wholesale on a $70 item) |
| Pure Powders | Flat $40, **off the volume curve**; subscribe + points may still apply down to a **per-SKU $30 floor** (75% of their list) |
| Points earning | Earn on **full-price spend only** (not on discounted/subscription orders); redeem above the floor |
| Floor representation | **Percent-of-list** (57% / 43%), per-SKU pct or absolute wholesale override allowed; all console-settable |
| Shipping | **Always charged on every order at actual USA cost** — never free, never an incentive. Consolidation savings reach the customer via lower actual shipping (format-driven). **Ship-to US addresses only** (international uses a US forwarder; we charge USA shipping only) |
| Tax treatment | Points = a **price discount** (reduces the GET tax base), not a tender |

---

## Part A — Discount & Points Stacking Engine

### A.1 The principle
Floors are the **safety net**, not the primary control. The primary control is **which
discounts may combine**. Most orders should *not* land on a floor.

### A.2 Discount buckets and additivity
**At most one % discount total** = the deepest single candidate (best-of-one), then points.

| Candidate | Source | Rule |
|---|---|---|
| Volume | months-based curve (§A.7) | best-of |
| Subscriber tier | 5/10/15% by order_count | best-of; exclusive of coupons |
| Coupon | `coupons.json` daily/seasonal | best-of; one only |
| Points | loyalty redemption ($ off) | applies after the % discount |

**Best-of-one (locked):** the order's single percentage discount =
`max(volume_pct, subscriber_tier_pct_or_coupon_pct)`. Subscriber tier and coupon never
stack with each other; volume is compared against whichever of those applies and the
deepest wins. Points then apply on top of whatever % won.

- Subscription order → % = `max(volume, subscriber tier)`. Coupons ignored.
- One-time order, member → % = `max(volume, member discount)`.
- One-time order, non-member → % = `max(volume, best coupon)`.
- (Open default, see §F: member-price vs coupon also follows best-one-wins.)

Because volume is **just another % candidate**, the engine's base stays the **true
single-unit list** (not a volume-reduced price), so floors always anchor to list and the
old "floor of a volume price" margin leak cannot occur.

### A.3 Floors (percent of list, console-settable)
- **Global defaults** (apply to every SKU unless overridden): `discount_floor_pct = 0.57`
  → price after % discount clamps **up** to `list × 0.57` (= wholesale; $40 on $70);
  `points_floor_pct = 0.43` → price after points clamps **up** to `list × 0.43` ($30 on $70).
  Confirmed by Glen (2026-06-13): these defaults cover **all Functional Formulations™**.
  **Pure Powders** ($40 list) carry a per-SKU floor override at **$30 (75% of list)** for
  both discount and points, and are excluded from the volume curve (§A.7).
- **Per-SKU floor override (console-editable, no deploy):** for higher-cost products found
  on review, a product can carry a **higher floor** than the global. Stored as per-SKU
  fields on the product record — either override percentages
  (`sku_discount_floor_pct` / `sku_points_floor_pct`, e.g. 0.70 / 0.60) **or** an absolute
  `wholesale_cents` (from which the points floor = `wholesale_cents − points_allowance_cents`).
  When set, that SKU's floors use the override; otherwise the global pct. Added/edited in the
  **Products console** (new per-product floor fields) so Glen can raise a floor on any item
  he flags without a code change. The pricing engine reads the per-SKU value first, global
  default second.

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
`pricing.compute(items, *, settings, subscriber_tier_pct=None, coupon_pct=None,
points_to_redeem_cents=0, channel, ship_to_state, tax_fn) -> dict`. It first totals the
**eligible months** across the cart, derives `volume_pct` from the curve (§A.7), then per
line applies `max(volume_pct, subscriber_or_coupon_pct)` to the true list, clamps to the
discount floor, subtracts allocated points, clamps to the points floor, and sums GET tax on
the discounted subtotal. Returns per-line breakdown + order subtotal, discount_cents,
points_redeemed_cents, get_cents, total_cents. **Used by all three callers** — cart preview,
one-time/reorder checkout, and the subscription scheduler — so they can never diverge.
Shipping is added by the caller at checkout (§A.8), not by the engine.

### A.7 Volume / quantity pricing (months-based, mix-and-match)
- **Unit = months** (bottle-equivalents): a standard 30-cap bottle = 1 month; larger
  formats count their multiple (90=3, 180=6, 360=12); each FF powder/liquid bottle = 1.
- **Mix-and-match:** sum months across **all volume-eligible lines** (all Functional
  Formulations; Pure Powders and `info_only` items excluded) → one total `M`.
- **Curve:** `volume_pct(M)` interpolates a console-tunable anchor table — default knots
  `{1:0%, 3:14%, 6:29%, 12:43%}` off (i.e. 100/86/71/57% of list), flat at 43% beyond 12.
  Front-loaded, monotonic total cost. `volume_pct` is the same for every eligible line
  (flat-at-total-level), then enters the best-of-one of §A.2.
- **Format is decoupled from price:** capsules may be fulfilled as one larger bottle,
  separate bottles, or cellophane refill packs at the **same price**; powders/liquids are
  individual bottles only. Format only changes **shipping** (§A.8), which the customer
  captures as a real saving.

### A.8 Shipping (always charged)
- **Every order is charged shipping at actual USA cost** (existing USPS flat-rate + box-fit
  at `/admin/shipping`). Never free, never used as an incentive (not for volume, subscribe,
  or points). Consolidation (one larger bottle vs many) lowers the real parcel cost, and
  that saving reaches the customer directly.
- **Ship-to US addresses only.** International customers enter a US forwarder address; we
  charge **USA shipping only** and never quote international postage. Cart shows a note.
  Checkout validates a US ship-to state (drives GET too).

### A.9 Rewards tiers & cash-out (earn/payout layer — Plan 2/3 scope)
Three referrer/earner tiers, by relationship (tier read from People-hub / GHL tags):
- **Client affiliates** (customers who refer) → earn **points** on referred sales, applied
  to their own purchases. Self-serve.
- **Doctors / practitioners** → earn **points** applied to their purchases; eligible for
  cash-payout review (below).
- **Professional influencers** → **cash commission**, individually approved by
  application/negotiation (the existing affiliate/commission track, not points).

Earn rules:
- Own purchases: 5% loyalty points on **full-price** spend; **suppressed on an
  affiliate-acquired first order** (the affiliate owns that acquisition — decision (b)),
  earned on every order after.
- Referral: the referrer (client affiliate or doctor) earns referral points when their
  referred buyer purchases.

Points value asymmetry (deliberate liability lever):
- **Redeemed on product/service = 100% face value; cash-out = a haircut** (default ~70% of
  face, console-set) — points are worth more kept in the ecosystem.
- Cash-out is **not automatic**: when a balance crosses a console threshold it raises a
  **payout-review task** (Business OS action/Tasks spine) for Glen/Rae to approve; capture
  W-9 / 1099 tracking there.
- All redemption (incl. doctors buying at wholesale) stays **bounded by the floors**.

The `points_ledger` (Task 6) is the routing-agnostic balance store under all of this; the
tiers, referral attribution, cash-out rate, and threshold review are the earn/payout layer
in Plan 2/3.

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
**Global:** `discount_floor_pct` (0.57), `points_floor_pct` (0.43), `points_earn_pct`
(0.05), `points_redeem_per_point_cents`, `points_allowance_cents`, `subscribe_tiers`
([5,10,15]), `cadences` ([1,2,3]), `lead_days` (3), `dunning_schedule` ([1,3,5]).
**Per-SKU (Products console):** `sku_discount_floor_pct`, `sku_points_floor_pct`, and/or
absolute `wholesale_cents` — raise the floor on a flagged high-cost product without a deploy.

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
  charge; pricing engine + global floors + **per-SKU floor override fields in the Products
  console** + points ledger; heads-up + receipts + dunning; manage portal
  (skip/pause/cadence/cancel) + Stripe card portal; consent checkbox.
- **v2:** edit items mid-subscription; annual prepay; biofield-analysis-driven product swaps;
  auto-points-redeem on subscription; gift / multi-address.
