# Sellable Program/Bundle SKUs with Bundle-Scoped Autoship — Design

**Date:** 2026-07-16
**Status:** Draft for review
**Repo:** deploy-chat

## Goal

Bring every multi-product **program/bundle** currently sold on remedymatch.com into
the deploy-chat catalog (`data/products.json`) as a **sellable, subscribable bundle
SKU**, with:

1. **One-time price = 10% off the summed retail price of its components**, computed by
   rule (not hand-set).
2. An **autoship** option using a new **bundle-scoped escalating discount ladder,
   12% → 29%** ("Save more each time"). The existing single-SKU autoship ladder
   (3% → 25%) is unchanged.
3. **Device bundles do not offer autoship** (one-time purchase only).

## Background (what already exists)

- **Autoship is a mature, in-production system** — `dashboard/subscriptions.py`
  (`subscriptions` table, Stripe card vaulting), checkout at `POST /reorder/subscribe`
  (app.py), monthly `POST /api/cron/charge-subscriptions`, and a self-manage portal
  (`/subscription`). The subscriber loyalty ladder is
  `SUBSCRIBE_TIERS = [3,5,7,…,25]`, applied via
  `pricing.compute(..., subscriber_tier_pct=…)`. **We reuse all of it.**
- **Bundles already exist** as a first-class catalog concept: a product with
  `bundle: true` + `bundle_components` (component **names**). Three live bundle SKUs
  exist (`dry-eye-relief-program`, `glucose-tolerance-program`, `macular-wellness-program`).
  Today a bundle's `price_cents` is **hand-set**; components are used only for
  shipping/packing (`dashboard/shipping.py::bundle_component_products`).
- **The gap:** bundles were never joined to autoship, and their prices are not
  rule-derived. Several programs on remedymatch.com aren't in the catalog at all.

## Scope — this phase

Port the **10 bundles whose components all resolve to priced catalog SKUs.** Defer 4
whose components include SKUs not yet in the catalog.

### Port now — 8 remedy bundles, WITH autoship (12→29)

| Bundle | Catalog action | Components | Retail sum | New one-time (10% off) | Prev price |
|---|---|---|---|---|---|
| Crystalline Lens Program | **new SKU** | 5 | $349.85 | **$314.86** | $279.97 (live) |
| Gut Terrain Program | upgrade plain→bundle | 3 | $209.91 | **$188.92** | $159.97 |
| Dry Eye Relief Program | re-price existing bundle | 3 | $209.91 | **$188.92** | $249.97 |
| Macular Wellness Program | re-price existing bundle | 5 | $349.85 | **$314.86** | $389.97 |
| Glucose Tolerance Program | re-price existing bundle | 2 | $139.94 | **$125.95** | $119.97 |
| Brain Program | upgrade plain→bundle | 5 | $349.85 | **$314.86** | $299.97 |
| Scar Reduction Program | **new SKU** | 4 | $249.88 | **$224.89** | $239.97 |
| IOP Program | upgrade plain→bundle | 5 (3×IOP Syntropy, 2×OcuFlow Daytime) | $349.85 | **$314.86** | $299.85 |

### Port now — 2 device bundles, ONE-TIME ONLY (no autoship)

| Bundle | Catalog action | Components (device in **bold**) | Retail sum | New one-time (10% off) | Prev price |
|---|---|---|---|---|---|
| Dental Bundle | **new SKU** | Dental Powder, **Wicking Toothbrush ×2** | $109.91 | **$98.92** | $79.97 |
| Sleep Bundle | **new SKU** | Brain Cleanse, Sleep Synergy, **Therapeutic Nightlight**, **Biocompatible Nightlight** | $389.94 | **$350.95** | $299.97 |

### Deferred — missing component SKUs (later phase)

| Bundle | Missing component(s) |
|---|---|
| OcuFlow Program | Neuro Magnesium Drink Mix |
| Reverse Aging Program | Vitamin C Syntropy |
| Skin Bundle | MSM Lotion |
| Travel Bundle | Ionizer-Wearable, Q-Link, Molecular Hydrogen bottle, Binder (devices) |

**Money note:** re-pricing the 3 existing bundles is deliberate. Dry Eye (−$61.05) and
Macular (−$75.11) currently cost **more** than buying their bottles à la carte
(+~$40 over retail sum); the rule corrects that inversion so a bundle costs less than
its parts. All component prices verified as explicit $69.97 (not defaulted).

## Design

### 1. Data model (`data/products.json`)

Each ported bundle SKU gains/keeps:

```jsonc
"crystalline-lens-program": {
  "name": "Crystalline Lens Program",
  "bundle": true,
  "bundle_components": ["Crystalline Clarity", "Clear Lens Eye Drops", "Clarity", "Golden Book", "Crucifer Complex"],
  "bundle_component_slugs": [                    // NEW: money-path resolution by slug
    {"slug": "crystalline-clarity", "qty": 1},
    {"slug": "clear-lens-eye-drops-aces-cat-eye-drops", "qty": 1},
    {"slug": "clarity", "qty": 1},
    {"slug": "golden-book", "qty": 1},
    {"slug": "crucifer-complex", "qty": 1}
  ],
  "price_rule": "components_less_10pct",         // NEW: marks rule-computed price
  "price_cents": 31486,                          // computed = round(0.9 * Σ component price_cents*qty)
  "autoship_eligible": true,                     // NEW: false for device bundles
  "bundle_description": "…"
}
```

- **`bundle_component_slugs`** is added because the money path must not depend on the
  fragile case-insensitive **name** match (`bundle_components` stays for
  shipping/packing back-compat). Slugs resolve through the canonical `_get_product`
  (follows `superseded_by`, drops `inactive`).
- **`autoship_eligible`** defaults to `true`; set `false` on device bundles. Rule for
  classification: **a bundle is autoship-eligible only if every component is a
  consumable remedy; any durable device makes it one-time-only.** (Dental, Sleep, and
  the deferred Travel are device bundles.)

### 2. Rule-computed pricing (the "not set by hand" guarantee)

- **Build script** `scripts/compute_bundle_prices.py`: for every product with
  `price_rule == "components_less_10pct"`, resolve `bundle_component_slugs` via
  `_get_product`, compute `price_cents = round(0.9 * Σ price_cents*qty)`, and write it
  back. Idempotent; run whenever a component price changes.
- **Drift-guard test** (pytest): for every `components_less_10pct` bundle, assert the
  stored `price_cents` equals the freshly computed rule value. Fails loudly if a price
  is hand-edited or a component price moves without a recompute. This is what makes the
  price rule-enforced rather than hand-set.
- Runtime pricing is unchanged: `_price_cart` still reads the bundle's static
  `price_cents` for the single bundle line and expands components for packing. (No live
  computation in the hot path.)

### 3. Bundle-scoped autoship ladder (12 → 29), applied **per line**

- **New ladder** in `dashboard/subscriptions.py`:
  `BUNDLE_SUBSCRIBE_TIERS = [12,14,16,18,20,22,24,26,28,29]` (escalates +2, caps 29),
  with `tier_for_bundle(order_count)` mirroring `tier_for`.
- **Per-line selection (not per-subscription).** Each subscription line earns the
  ladder appropriate to *what that line is*, using the subscription's `order_count`:
  - a **bundle** line (`autoship_eligible` bundle SKU) → `tier_for_bundle(order_count)`
  - any **other** line (single SKU) → `tier_for(order_count)` (unchanged 3→25)
  - So a subscription mixing a program bundle + a loose bottle discounts the bundle at
    12→29 and the bottle at 3→25, in the same charge.
- **Pricing-engine change (the core work).** `pricing.compute` today applies one scalar
  `subscriber_tier_pct` to the whole cart. Extend it to resolve the subscriber tier
  **per item**: for each line, pick the tier via a small resolver
  `subscriber_tier_pct_for(item, order_count)` = `tier_for_bundle` if the item is an
  autoship-eligible bundle else `tier_for`. The existing single-scalar call sites keep
  working (a scalar still applies uniformly); the subscribe path and the charge-cron
  switch to passing the per-item resolver.
- **No schema change.** `order_count` already lives on the `subscriptions` row. A
  pre-existing single-SKU subscription has only non-bundle lines → identical discount
  to today. **No live subscription changes behavior.**
- **Charge-cron** (`/api/cron/charge-subscriptions`) and **`/reorder/subscribe`** both
  re-price via the per-item resolver instead of a single `subscriber_tier_pct`.
- **Device gate:** `/reorder/subscribe` **rejects** a cart whose bundle line has
  `autoship_eligible == false` (400 with a clear message); the product page hides the
  subscribe CTA for those bundles.

**Money-path risk:** this touches the pricing engine's subscriber-tier application.
Guard with characterization tests proving (a) a single-scalar call is unchanged, (b) a
single-SKU subscription charges exactly as before, (c) a bundle line gets 12→29, and
(d) a mixed cart splits correctly.

### 4. Surfaces

- **Each ported bundle gets a new product page on the new storefront** (the deploy-chat
  site), and the SKU's `url` points to that page — not the remedymatch.com listing.
- The page shows the one-time price and, when `autoship_eligible`, a "**Subscribe &
  save — save more each time**" CTA that opens the existing subscribe flow. Device
  bundles (Dental, Sleep) show one-time only.
- No change to the subscription self-manage portal or Stripe/QBO plumbing.

## Non-goals

- Not changing the global single-SKU ladder (`SUBSCRIBE_TIERS`, 3→25).
- Not building the 4 deferred bundles (needs their component SKUs, incl. device pricing).
- Not wiring the eye `condition_programs_seed.json` protocols to these bundle SKUs
  (they don't map 1:1 to the store programs — possible follow-up).

## Testing

- **Pricing drift-guard** test (all `components_less_10pct` bundles).
- **Ladder** test: `tier_for_bundle` values; `tier_profile` selection for
  bundle-only / single-SKU / mixed carts.
- **Device gate** test: `/reorder/subscribe` rejects an `autoship_eligible=false`
  bundle; accepts an eligible one.
- **Cron** test: a `tier_profile='bundle'` subscription charges at the 12→29 ladder;
  a single-SKU line in the same subscription still charges at 3→25 (per-line split).
- **Component-resolution** test: every ported bundle's `bundle_component_slugs`
  resolve via `_get_product` (no unknown slugs).
- **End-to-end verify:** render a ported bundle page, confirm one-time price + CTA,
  drive a subscribe, confirm the `subscriptions` row + `tier_profile`.

## Rollout / money safety

- deploy-chat is merge=deploy (no CI) — verify live after deploy.
- Re-pricing the 3 existing bundles changes live prices (deltas above) — intended.
- Device bundles ship with no autoship path.
- Build script + drift-guard run before deploy so no bundle ships with a stale price.

## Resolved decisions (from review)

1. **Bundle ladder shape** — `[12,14,16,18,20,22,24,26,28,29]` (steps of 2, cap 29). ✅
2. **Device classification** — Dental (toothbrush) + Sleep (nightlights) are device
   bundles → one-time only, no autoship. ✅
3. **Mixed-cart tier** — **per line**: the bundle line earns 12→29, the single SKU
   stays on 3→25, within the same subscription. ✅
4. **Bundle `url`** — each ported bundle gets a **new product page on the new
   storefront**; the `url` points there. ✅
