# Product Discount Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes. Branch: `discount-controls` (off main). Spec: `docs/superpowers/specs/2026-07-02-discount-controls-design.md`. Test cmd (cwd resets → cd each call): `cd /tmp/wt-deploy-chat-0cdf9c99 && doppler run -p remedy-match -c dev -- python3 -m pytest tests/<file> -q`.

**Goal:** Replace the single open-to-all `volume_anchors` discount with three independently console-toggleable discount TYPES, resolved NON-ADDITIVELY (each line charged the single lowest price = highest applicable %), each a LINEAR 2-anchor ramp:
- **Type 1 same-SKU** (`same_sku`): keyed on that line's own SKU `qty`; open to everyone; default ON `[[1,0],[12,29]]`.
- **Type 2 program order-total** (`program_total`): keyed on order-total (`total_months`); gated on `_is_paid_member(email)` (passed as `program_member` bool); default ON `[[1,0],[12,29]]`.
- **Type 3 open order-total** (`open_total`): order-total, everyone; **default OFF/zero** `[[1,0],[12,0]]`; inherits legacy `volume_anchors` when a saved override exists.

Selection per line: `line_pct = max(type1, type2, type3, subscriber/coupon)`, clamped at `discount_floor_pct` (extends engine's existing max-not-additive model).

## Architecture
Pure engine `dashboard/pricing.py` stays DB-free; membership passed in via new `compute(..., program_member=False)` param (like `subscriber_tier_pct`). New pure helpers `_ramp_pct`, `_discount_cfg`, `same_sku_pct`, `program_total_pct`, `open_total_pct`. Settings `dashboard/pricing_settings.py` gains a `discounts` block (`defaults_view`/`effective`/`validate`, reuse `_validate_anchors`); legacy `volume_anchors` feeds `open_total.anchors` when the new block is absent. Callers in `app.py` thread `program_member=_is_paid_member(email)`. Console `static/console-pricing-settings.html` gains three enable-toggle + ramp-editor blocks.

## Global Constraints
- `discount_floor_pct=0.57`, `points_floor_pct=0.43` (unchanged clamps).
- Type-1 keys on engine item `qty` (raw SKU count); type-2/3 on `total_months = sum(int(it["months"]) for it in items if it["volume_eligible"])`.
- Only `volume_eligible` lines get type-1/2/3; `base_pct` (subscriber/coupon) always applies.
- Type-2 gate = `_is_paid_member(email)` (ASSUMPTION — owner leaned "active care/program member"; this uses the existing paid-membership grant, excludes unconverted $1-trial; the $100/$300 sub-project-3 program wires in later behind the same bool).
- Legacy `volume_pct(months, settings)` (reads `volume_anchors`) KEPT unchanged — still backs the OWNER in-house order-total rate. Only the customer-facing `compute` cart moves to the three-type engine.
- Default `discounts` block: `{"same_sku":{"enabled":True,"anchors":[[1,0],[12,29]]}, "program_total":{"enabled":True,"anchors":[[1,0],[12,29]]}, "open_total":{"enabled":False,"anchors":[[1,0],[12,0]]}}`.

## Owner-decision notes (flag; don't re-litigate mid-build)
1. Type-2 gate = `_is_paid_member(email)` (existing paid membership; NOT yet the $100/$300 program tiers — those wire in later behind the same `program_member` bool). DECISION: decouple, ship on the existing paid-membership grant now.
2. Deploy-safety = **Task 3 (type-1 ON + type-3 OFF together)**, not type-3-off alone — otherwise legit single-SKU discounts drop to zero. Tasks 1–2 are pure additive scaffolding, zero live-behavior change.
3. OWNER in-house path + product-page single-SKU tiers keep `volume_pct(volume_anchors)` (owner-controlled, not public mix-and-match). Only customer-facing `compute` cart moves to the three-type engine.
4. Legacy back-compat: a saved prod override with only top-level `volume_anchors` auto-populates `open_total.anchors` (still `enabled=False`); `same_sku`/`program_total` fall to code defaults. At go-live confirm `open_total.enabled=false` + the desired single-SKU curve in `same_sku`.

## Tasks
(Full per-step real code + tests are in the plan-agent transcript 2026-07-02 a1085eed; reproduce verbatim. Summary + interfaces below.)

**Task 1 — `discounts` settings shape** (`pricing.py` DEFAULTS + `pricing_settings.py` defaults_view/effective/validate + `tests/test_pricing_settings.py`). No live-behavior change. Add the `discounts` DEFAULTS block; `defaults_view` deep-copies it; `effective` synthesizes it w/ legacy `volume_anchors`→`open_total.anchors` (disabled); `validate` validates each type (bool `enabled` + `_validate_anchors(anchors,1)`). Tests: defaults present + deep-copy, effective back-compat, validate accept/reject.

**Task 2 — pure ramp helpers** (`pricing.py` + `tests/test_pricing_engine.py`). No live change. `_ramp_pct(qty,anchors)` (generalize the interp), refactor `volume_pct` to call it (behavior identical); add `_discount_cfg(settings)` (back-compat mirror), `same_sku_pct(line_qty,settings)`, `program_total_pct(total_qty,settings,program_member)`, `open_total_pct(total_qty,settings)` — each returns 0 when its type disabled (program_total also 0 when not member). Tests: each helper's on/off/gated/linear behavior.

**Task 3 — non-additive selection in `compute` (DEPLOY-SAFETY milestone; LIVE change)** (`pricing.py` + `tests/test_pricing_engine.py`). Add `program_member=False` to `compute`. Compute `open_pct=open_total_pct(total_months,settings)`, `prog_pct=program_total_pct(total_months,settings,program_member)`; per line (eligible only) `t1=same_sku_pct(qty,settings)`, `order_pct=max(prog_pct,open_pct)`, `line_pct=max(t1,order_pct,base_pct)`. Keep `volume_months`; set result `volume_pct=max(prog_pct,open_pct)`. Realign the old mix-and-match test → now requires program_member (guest gets subscriber/same-sku, member gets order-total 29%); add single-SKU-open-to-all + open_total-when-enabled tests. Then confirm `test_price_cart`/`test_begin_checkout_engine`/`test_reorder_checkout_engine` still green (single-SKU qty6 → type1 13.18% → discount 5536).

**Task 4 — thread `program_member` through callers** (`app.py` + `tests/test_price_cart.py`/`test_begin_checkout_engine.py`). Add `program_member=False` to `_price_cart` signature; pass into `compute`. Set at email-bearing sites: `begin_checkout`, `_checkout_cart`, subscription-ship + founding paths = `_is_paid_member(email)`; `api_pricing_preview` = `bool(data.get("program_member"))`; `_price_biofield` + no-email paths = False. Tests: member vs guest mix-cart pct_applied.

**Task 5 — product-page tiers → type-1; owner in-house unchanged** (`app.py` `begin_product_data` + tests). Swap the product-data tier source `volume_pct(m,_s)` → `same_sku_pct(m,_s)` (single product = same-SKU; numerics identical to default → values unchanged). Confirm `test_inhouse_volume_pricing` (still on `volume_pct`) + `test_product_data_volume_tiers` green.

**Task 6 — console UI three-block editor** (`static/console-pricing-settings.html` + `tests/test_console_pricing_settings_routes.py`). Three enable-checkbox + ramp-table blocks (`ds_<type>_enabled` / `ds_<type>_anchors`), `readDiscounts()`/`readDiscountRamp()`, extend `populate(eff)` + `buildPayload()` with `discounts`. Copy note on `open_total`: "Off by default — conflicts with remedymatch.com public store." Tests: POST persists discounts round-trip + page has the three tokens.

## Final gate (no commit)
Run all: test_pricing_engine, test_pricing_settings, test_console_pricing_settings_routes, test_price_cart, test_inhouse_volume_pricing, test_product_data_volume_tiers, test_begin_checkout_engine, test_reorder_checkout_engine → all PASS. Confirm deploy-safety by hand: `compute(<mix cart>, settings=load_settings({}), program_member=False)` → line pct = same_sku only (order-total types 0); fresh deploy (no pricing-settings.json) = type-3 OFF = no public-store mix-and-match discount. Then full-suite before/after `comm -23` diff vs merge-base = zero net-new (pollution-aware).
