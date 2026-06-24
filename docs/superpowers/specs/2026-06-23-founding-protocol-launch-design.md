# Founding Protocol — Product-Led Founding Launch (Design Spec)

**Date:** 2026-06-23
**Status:** Design — pending Glen's review
**Repo:** deploy-chat (illtowell.com funnel)
**Related:** `project_neuro_magnesium_launch`, `project_membership_pause_loyalty`, `project_studio_credit_free_month`, `project_jon_benson_video_process`

---

## 1. Goal

Stand up the **first real illtowell.com product launch** — Neuro Magnesium — as a **product-led founding offer**, and build it as a **repeatable launch machine** that each future outsourced product re-runs.

The offer (founding cohort only — capped at the first 2,500-bottle production run):

> **Get your first Neuro Magnesium (~$80 + shipping). Your first month of membership — live group coaching + premium access + community ($99 value) — is on us. Stay on monthly autoship and your membership stays free for as long as you're on protocol. One-click cancel, anytime.**

### Strategic logic (why product-led, not membership-led)
- The **bottle is the recurring revenue** (~$62 margin/mo); the **membership is the free loyalty layer** that suppresses churn (live-group-coaching + community: ~3–6%/mo vs self-serve 8–30%).
- A consumable is naturally re-bought; a membership is naturally cancelled. Put the recurring charge on the thing people instinctively re-up.
- Sidesteps the unproven "will AMD patients pay $99/mo for coaching" assumption — they pay for the bottle (proven category willingness); coaching rides free.
- The free membership is a **founding-launch-only** acquisition lever (capped to the 2,500 cohort, then it closes) — the disciplined way to give the membership away without it becoming permanent pricing.

## 2. Scope

**In scope (this spec):** the offer mechanics + how they wire into the existing illtowell funnel, the Neuro Magnesium product page (with the v1 promo video), the founding-cohort cap/counter, and the pre-order/charge-on-ship flow. Parameterized so future products re-run it.

**Out of scope (explicitly):**
- The **acquisition engine** (host-beneficiary partners, free book/quiz lead magnet, "save your sight" webinar, AMD nurture) → separate spec, built after.
- **Standard post-founding pricing** (what non-founders pay once the 2,500 sell) → a later decision; not required to launch the founding cohort.
- Manufacturing/procurement (the PureNSM order) — a real-world dependency tracked in `project_neuro_magnesium_launch`, not a code dependency. Pre-order is designed precisely to precede it.

## 3. Existing architecture this builds on (confirmed)

| Piece | Where | Reuse |
|---|---|---|
| Product catalog | `data/products.json` | Neuro Magnesium entry already exists; set price + content |
| Product/buy pages | `/begin/product/<slug>`, `/begin/buy/<slug>`, `/begin/product-data/<slug>`; `static/begin-product.html`, `begin-buy.html` | Slug-agnostic; populate + add video |
| Checkout | `/reorder/checkout` → `dashboard.pricing.compute()` → Stripe → `_ingest_order(source=...)` → `orders` | Reuse; add founding source/flag |
| Autoship / card-vault | `subscriptions` table (vaulted `stripe_payment_method_id`, `cadence_months`, `next_charge_date`, `order_count`, `skip_next`); off-session charge cron | The charge-on-ship + monthly autoship engine |
| Membership grant | `_grant_membership(cx,email,days,source)`, `_extend_membership_grant(cx,email,until_iso,source)`, `_active_membership_for_email(email)` | Comp-grant the free membership |
| Coaching window | `_open_coaching_for_order(...)`, `_activate_coaching_for_shipment(...)` (delivery → 30-day window) | Already fires on bottle delivery |

## 4. The two new linkages (the actual build)

### 4a. Membership-free-while-product-autoship-is-active
New rule: an **active founding product autoship** grants/extends a **comp membership** (not a paid $99 charge).
- On founding signup: `_grant_membership(email, 30, source="founding")`.
- On each successful autoship charge (in the charge cron's success branch): `_extend_membership_grant(email, next_charge_date + grace, source="founding")` — same monotonic extend the paid path uses, but triggered by the *product* charge.
- On autoship cancel/lapse: membership grant simply isn't extended → access lapses naturally (existing behavior). No new teardown code.
- **Invariant:** this comp membership is scoped to founders (a `founding` flag); standard members still pay $99/mo. The free membership is a property of *being an active founding autoship subscriber*.

### 4b. Pre-order / charge-on-ship founding reservation
Founders reserve before the batch is produced.
- Reuse the subscription card-vault pattern (Stripe setup-mode): vault the card now, **charge $0 today**.
- Create the subscription row with the bottle item + cadence 1 month, but **first charge fires when the founding batch ships** (a `founding_pending` state; first charge triggered by the ship event / an admin "ship founding batch" action), not on a fixed future date.
- After the first on-ship charge, the autoship proceeds monthly as normal.
- A founder who reserved counts toward the cap immediately (card vaulted), so the counter reflects committed demand.

## 5. Founding-cohort mechanics
- **Cap = 2,500** (config per launch). A live **reservation counter** ("1,847 of 2,500 claimed") on the product/landing surface = the scarcity engine (honest scarcity: real batch size).
- Each founding reservation gets a **founding flag** (and optional sequence number) on its `orders`/`subscriptions` row → grandfathers the free-membership-while-subscribed deal.
- When the cap is hit (or the launch window closes), the founding offer **closes**: the surface flips to a waitlist/"Batch No. 2 coming" capture and standard terms.
- "Limited founding members per coaching cohort because coaching is live and hands-on" is a second honest-scarcity message tied to coaching throughput.

## 6. Parameterization (the repeatable machine)
A **founding-launch config** (per product slug): `{slug, cap, batch_label, founding_blurb, video_url, closes_at?}`. Launch #2/#3 add a config entry + a product page + a promo video (HyperFrames pipeline) and re-run the same flows. No new offer code per product.

## 7. Product page (Neuro Magnesium)
- Populate `products.json` Neuro Magnesium: name, price (~$80), description, ingredients/mechanism (ATA Mg / blood-brain + blood-eye barrier — structure-function language), benefits.
- Embed the **v1 promo video** (`~/Downloads/neuro-magnesium-promo-v1.mp4`, host on R2/clip serving) on `begin-product.html`.

## 8. Compliance (hard requirements)
- **Structure-function only** on every public surface; never state or imply the product treats/prevents/slows/reverses AMD/macular degeneration/glaucoma. No disease nouns as the thing the product acts on.
- **Founder reversal story = biography only** (2023 FTC Endorsement Guides treat a founder testimonial as a direct claim) — no "you will reverse too," disclose material connection, carry the DSHEA disclaimer.
- **Autoship under FTC ROSCA:** clear terms, express informed consent before the first (on-ship) charge, cancellation as easy as signup (one-click).
- Reuse the funnel's existing compliance denylist/guardrails.

## 9. Success criteria
- A visitor can reserve a founding Neuro Magnesium bottle (card vaulted, $0 today), see the live counter, and receive the free membership grant immediately.
- On founding-batch ship: first charge fires, autoship begins monthly, delivery opens the coaching window.
- An active founding autoship keeps membership access; cancelling lapses it; cancel is one-click.
- The flow is driven by a per-product config (provably re-runnable for product #2).
- Zero disease claims on any surface; ROSCA-compliant autoship disclosure.

## 10. Deferred to the implementation plan
- Exact charge-on-ship trigger (admin "ship founding batch" action vs shipment webhook) and `founding_pending` state machine.
- Counter storage (derive from `orders`/`subscriptions` count vs a cached value).
- Whether the standalone $99/mo membership door is surfaced now or later.
- Slicing (e.g., product page → pre-order/vault → free-membership linkage → cap/counter → close-out).
