# Product Discount Controls — Design Spec

**Date:** 2026-07-02 · **Status:** design for review · From Glen + Rae.

## Context / why
Rae + Glen want product discounting under **console control**: three independently-toggleable discount TYPES, each on/off + amount-adjustable, resolved **non-additively** (the customer is charged the single **lowest-price** offer, not a sum of discounts), with **linear** quantity ramps. One type (open-to-all, mix-and-match) conflicts with the public **remedymatch.com (GrooveKart)** store and must **default OFF**. Building this also makes the already-merged #489 pricing **safe to deploy** (today its open-to-all discount is active + not toggle-able).

## The three discount types
| # | Type | Keyed on | Who gets it | Default |
|---|------|----------|-------------|---------|
| 1 | **Same-SKU quantity** | that LINE's qty of one SKU (per-line) | everyone | ON *(amount TBD)* |
| 2 | **Program order-total** | TOTAL qty across the order | buyers tied to a paid **$100/$300 program** service fee | ON *(gating TBD — see decisions)* |
| 3 | **Open order-total (mix & match)** | TOTAL qty across the order | anyone | **OFF / zero** *(remedymatch.com conflict)* |

- Each type has its own **linear ramp** (2-anchor `[[qty,0],[qty_max,max_pct]]`) — 0% at qty 1 rising evenly to `max_pct` at `qty_max`, flat beyond. Console-adjustable per type.
- Type 1 keys on the **line's own SKU quantity**; types 2 & 3 key on the **order-total quantity** (mix-and-match across all eligible lines).

## Selection rule (non-additive — lowest price wins)
For each line: compute the candidate % from every **enabled + applicable** type, take the **max %** (= lowest price), then apply and clamp at the wholesale floor (`discount_floor_pct`). This extends the engine's existing `line_pct = max(volume, subscriber/coupon)` model — the code is already max-not-additive.

## How it maps to the current engine
- Today `dashboard/pricing.py` has ONE `volume_anchors` = an order-total, open-to-all, linear ramp. **That is exactly Type 3.** Restructure settings so:
  - `type3` (open order-total) inherits today's `volume_anchors`, **default enabled=false / zero**.
  - `type2` (program order-total) = same order-total mechanic, **gated** (see decisions), own ramp + enable.
  - `type1` (same-SKU) = a NEW per-line ramp keyed on the line's qty, own enable.
- `pricing.compute` gains: a per-line type-1 pct (from line qty), a type-2 pct (order-total, only when the buyer/order qualifies), a type-3 pct (order-total, only when enabled). `line_pct = max(type1, type2, type3, subscriber/coupon)`, clamped at floor.
- Console: extend `pricing-settings.json` + `dashboard/pricing_settings.py` (`defaults_view`/`validate`/`effective`, reuse `_validate_anchors`) + `POST /api/console/pricing-settings` + `static/console-pricing-settings.html` — three blocks, each an enable toggle + a 2-row linear ramp editor.

## Settings shape (proposed)
```
"discounts": {
  "same_sku":       {"enabled": true,  "anchors": [[1,0],[12,X1]]},   # line qty
  "program_total":  {"enabled": true,  "anchors": [[1,0],[12,29]]},   # order total, gated
  "open_total":     {"enabled": false, "anchors": [[1,0],[12,0]]}     # order total, everyone — OFF
}
```
(Keep the legacy `volume_anchors` key readable for one release as the `open_total` source, or migrate it; back-compat handled in `effective()`.)

## Deploy-safety note
The default must resolve type-3 to **off/zero** so a fresh deploy (no console override saved) does not apply an open-to-all discount. If a console override with live anchors is already saved on prod, it persists — so at go-live, confirm the saved pricing-settings has `open_total.enabled=false` (or zero the legacy `volume_anchors`) before/at deploy.

## Owner decisions needed (confirm to finalize + plan)
1. **Type-2 gating** — "program associated with a $100/$300 service fee" means: **(a)** the buyer is an active program/care member (paid the service fee) → order-total volume on their remedy orders [my recommendation — programs and remedy orders are separate checkouts]; or **(b)** the current cart contains a program line. Which?
2. **Type-1 scope + default amount** — open to everyone (packing-based), default ON; what `max_pct` at qty 12 (e.g. same 29%, or smaller)? Confirm it does NOT also conflict with remedymatch.com same-SKU discounts.
3. **Type-2 default amount** — linear to `29%` at qty 12 (same as the current ramp), or a different max?
4. **Type-3** — confirmed OFF/zero default. Keep it editable (so it can be turned on later) — yes?
5. **Program-fee tiers** — type-2's "$100/$300" ties to sub-project 3's program tiers (not yet built). Does type-2 ship gated on the EXISTING care/membership grant now, with the $100/$300 program wiring following in sub-project 3? [Recommend yes — decouple.]

## Success criteria
- Each discount type independently toggles on/off + amount-adjusts in the console, applied next order (no redeploy).
- Non-additive: a line never receives more than the single best applicable discount.
- Fresh deploy default = type-3 off (no public-store conflict).
- Existing pricing tests stay green; new tests cover per-type ramps + the max-selection + the gating.
