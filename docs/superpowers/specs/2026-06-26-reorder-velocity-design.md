# Sales Velocity → Reorder Demand — Design Spec

**Date:** 2026-06-26
**Status:** Design (build now). Connects the merged **Product Sales Aggregation**
([[2026-06-25-product-sales-aggregation-fmp]]) into the existing reorder report
(FMP→app migration Phase 3c-3, [[project_fmp_to_app_migration]]).

## Goal

Auto-generate the reorder report's production **plan** from sales velocity, so reorder demand
reflects what's actually selling. Show each formulation's **3-month and 12-month velocity side by
side** (trend vs baseline); default the projected demand to 3-mo × a tunable horizon, with a toggle
to 12-mo or "max of both." Read-only suggestion — no draft POs.

## Current state (reuse-first)

- **`dashboard/reorder.py`** already does the math: `bom_demand(plan)` explodes a plan
  (`[{formulation_id, qty}]`) into ingredient demand (`dose × qty` via `formulation_items`);
  `reorder_report(plan, …)` computes `shortfall = par + demand − on_hand − on_order`. **Both stay
  unchanged** — this feature only *generates the plan*.
- **`product_sales`** (just built): `product_fmp_id, period 'YYYY-MM', units, revenue_cents, source`.
- **Join:** `product_sales.product_fmp_id` = `formulations.fmp_id` → `formulations.id`
  (the `formulation_id` `bom_demand` consumes). Products with no formulation row (services like
  "Biofield Analysis") have no recipe → **dropped** from the plan.

## Design

### New functions in `dashboard/reorder.py`

- **`product_velocity(db_path=None) -> {product_fmp_id: {"vel_3mo": float, "vel_12mo": float}}`** —
  average **units/month** over the trailing 3 and 12 months. The window is anchored to the **latest
  period present in `product_sales`** (`MAX(period)`), not wall-clock today, so a month-stale extract
  still produces sensible numbers. `vel_N = SUM(units in the N calendar months ending at the latest
  period) / N` (missing months count as 0 → standard average monthly velocity). Sums across all
  `source` values.
- **`velocity_plan(basis="3mo", horizon_months=3, db_path=None) -> [{formulation_id, qty}]`** —
  `vel = pick(basis ∈ {"3mo","12mo","max"})`; `qty = vel × horizon_months`; map
  `product_fmp_id → formulations.fmp_id → formulations.id`; **skip** rows with no formulation match or
  `qty <= 0`. The returned plan is exactly the shape `bom_demand`/`reorder_report` already accept.
- **`velocity_table(basis="3mo", horizon_months=3, db_path=None) -> [{formulation_id, fmp_id,
  name, vel_3mo, vel_12mo, projected_qty}]`** — per matched formulation, for the side-by-side console
  view (sorted by `projected_qty` desc).

### Endpoint + console

- Extend the existing reorder report endpoint (`/api/reorder/report`) to accept
  **`source=velocity&basis=3mo&horizon=3`**: when `source=velocity`, build the plan via
  `velocity_plan(basis, horizon)` and include the **`velocity_table`** in the response alongside the
  normal grouped-by-supplier reorder list (`reorder_report(plan)`). Console-key gated as today.
- The reorder console gets a **"From sales velocity"** mode: a **basis toggle (3mo / 12mo / max)** +
  a **horizon input** (defaults: 3-mo basis, 3-month horizon); it renders the velocity table
  (formulation · vel_3mo · vel_12mo · projected qty) and, below, the existing reorder shopping list.

## Out of scope

- Draft-PO creation (the reorder report stays read-only).
- Seasonality models beyond the 3-/12-mo averages; unit conversion (inherited from reorder.py).
- The manual-plan path is unchanged — velocity is an *additional* plan source, not a replacement.

## Dependencies

- `product_sales` must be populated (the FMP invoice import — Glen activates via the console). Until
  then `velocity_plan` returns `[]` and the report is empty (no error).
- The `fmp_id` join: only formulations that carry an `fmp_id` and have `formulation_items` produce
  demand.

## Testing (run via [[reference_deploy_chat_local_tests]])

- `product_velocity`: seed `product_sales` spanning >12 months for a product → assert `vel_3mo`
  (last 3 months / 3) and `vel_12mo` (last 12 / 12) computed relative to `MAX(period)`; a product
  selling in only 1 of the last 3 months → `vel_3mo = units/3` (partial window averaged over N).
- `velocity_plan`: `basis` selects the right velocity; `qty = vel × horizon`; `product_fmp_id`→
  `formulations.id` mapping (seed a formulations row); products with no formulation match or zero
  qty are dropped; `max` basis takes `max(vel_3mo, vel_12mo)`.
- integration: a `velocity_plan` fed to the existing `reorder_report` yields the expected ingredient
  shortfalls (seed `formulation_items` + an ingredient with par/on-hand).
- endpoint: `source=velocity` returns `{velocity_table, …reorder report…}`; console-auth (200 w/ key,
  401 without).

## Rollout

Additive — new functions in `reorder.py`, a `source=velocity` branch on the existing endpoint, and
the console mode. Console-gated; no public flag. Read-only.
