# Phase 3c-3 Spec — BOM Demand + Reorder List

## Context

The capstone of Phase 3 (and the whole FMP→app-DB migration). Everything upstream now exists in `chat_log.db`:
- **Recipes** — `formulations` + `formulation_items` (Phase 2): each line = ingredient + `dose` + `dose_unit`.
- **On-hand** — `inventory_txns` signed SUM (3c-1): baseline + receipts − consumption ± recounts.
- **On-order** — open `purchase_orders` (status `open`) → `po_items` ingredient lines not yet fully received (3b).
- **Par + preferred source** — `ingredients.extras.par_level` (3c-1) and `ingredient_sources` (Phase 1: `preferred`, `price_per_unit`, `unit_size`, `minimum_order`, supplier).

3c-3 turns all of it into the answer Glen actually wants: **"what do I need to buy?"** It explodes a planned production list through the recipes, nets the resulting demand (plus par as a floor) against on-hand + on-order, and produces a read-only reorder shopping list grouped by preferred supplier. **No new tables, no FMP import — pure read/compute over existing data.**

### The reorder formula

For each ingredient, the goal is: *after fulfilling the planned production, still be at or above par.*
```
shortfall = (par_level or 0) + demand − on_hand − on_order
reorder if shortfall > 0
```
- **No plan** (demand = 0) → `shortfall = par − on_hand − on_order` = pure reorder-from-par ("what's below par right now").
- **No par** (par = 0) → `shortfall = demand − on_hand − on_order` = pure demand-driven.
- **Both** → keep par on the shelf *and* cover the plan. Intuitive and unified — no `max()` ambiguity.

Suggested order qty rounds the shortfall up to the preferred source's `minimum_order` (and, if set, to `unit_size` multiples); estimated cost = suggested qty × `price_per_unit`.

### Units (deliberate, consistent with the whole migration)

Demand (recipe `dose_unit`), on-hand (ledger unit), par (`par_level_unit`), on-order (PO unit), MOQ (`minimum_order_unit`) are all free-text and **not converted** — the system has never done unit math (Glen's approximate-then-recount stance). 3c-3 sums numerically and **surfaces the unit strings** so the operator sanity-checks; where an ingredient's relevant unit strings disagree, the report flags a `unit_warning` (display only, never blocks). No g↔kg↔ea conversion in v1.

## Scope

**In:**
- `dashboard/reorder.py` — pure-compute module: `bom_demand` (explode a plan through recipes), `on_order_by_ingredient` (open-PO ingredient qty not yet received), `reorder_report` (the netted, grouped-by-supplier shopping list).
- `/api/reorder/*` endpoints (compute the report from an optional plan).
- A "Reorder" console tab: build an optional production plan (search-to-pick formulation + qty lines), toggle "include below-par", compute → grouped-by-supplier shopping list with per-supplier subtotals; read-only.

**Out (deferred):**
- **Draft PO creation** from the list — Glen's chosen output is a read-only shopping list (the app's never-auto-order posture). Forward PO creation is a later, separate piece.
- **Unit conversion / normalization** — surfaced as warnings, not resolved.
- **Saved/named production plans** — the plan is ad-hoc per compute (passed to the endpoint); persistence is YAGNI for v1.
- **Material reorder** — ingredient-only (materials inventory isn't tracked; consistent with 3c-1/3c-2).
- **Lead-time/lateness scheduling** — `lead_time_days` is shown but not used to time orders.

## Components & critical files

No schema changes. All functions take optional `db_path`, use `with _connect(db_path)` (reuse the `dashboard/inventory.py` pattern). Reuse `dashboard.inventory.on_hand`, `dashboard.formulations.list_items_for_formulation`.

1. **`dashboard/reorder.py`** (new — name confirmed collision-free):
   - `bom_demand(plan, db_path=None) -> dict` — `plan` = list of `{formulation_id, qty}`. For each line, multiply each recipe item's `dose` by `qty` and accumulate per ingredient. Returns `{ingredient_id: {"demand": float, "unit": <first dose_unit seen>, "units_seen": set/list}}`. Skips recipe items with no `ingredient_id`. A `qty` or `dose` that isn't numeric contributes 0 (defensive).
   - `on_order_by_ingredient(db_path=None) -> dict` — `{ingredient_id: {"on_order": float, "unit": ...}}` from `po_items pi JOIN purchase_orders po ON po.id=pi.po_id` WHERE `po.status != 'closed'` AND `pi.ingredient_id IS NOT NULL`, with `received = COALESCE((SELECT SUM(qty_received) FROM po_receiving r WHERE r.po_item_id = pi.id), 0)` and `on_order += max(0, pi.qty − received)`. (Computed from receiving, not the FMP `qty_left` calc field — same trust rationale as the inventory ledger.)
   - `reorder_report(plan=None, include_below_par=True, db_path=None) -> dict` — the core. Steps:
     1. `demand = bom_demand(plan or [], db_path)`; `onord = on_order_by_ingredient(db_path)`.
     2. Candidate ingredient set = ingredients that appear in `demand`, OR (if `include_below_par`) every ingredient with a numeric `par_level` in `extras`, OR appear in `onord`.
     3. For each candidate: `oh = on_hand(id)`; `oo = onord.get(id, 0)`; `dem = demand.get(id, 0)`; `par = numeric(extras.par_level) or 0`; `shortfall = par + dem − oh − oo`.
     4. Keep lines where `shortfall > 0`. Resolve the preferred source (`SELECT ... FROM ingredient_sources WHERE ingredient_id=? ORDER BY preferred DESC, price_per_unit LIMIT 1` — same ordering as `list_sources_for_ingredient`): `supplier_id`, `supplier_name`/company, `price_per_unit`, `unit_size`, `minimum_order`.
     5. `suggested_qty = _round_up_order(shortfall, minimum_order, unit_size)` (≥ shortfall, ≥ minimum_order if set, rounded up to a `unit_size` multiple if set). `est_cost = suggested_qty * price_per_unit` when price known.
     6. `unit_warning = True` when the ingredient's relevant unit strings (par/demand/on-hand-source/MOQ) that are present disagree.
     7. Group the lines by supplier (preferred source's supplier; ungrouped bucket "— no preferred source —" for ingredients lacking one). Return `{"groups": [{"supplier_id","supplier","lines":[...],"subtotal":<sum est_cost>}], "totals": {"lines": N, "est_cost": <sum>}, "plan_echo": plan}`.
   - `_round_up_order(shortfall, moq, unit_size)` — helper: start `q = max(shortfall, moq or 0)`; if `unit_size` is a positive number, `q = ceil(q / unit_size) * unit_size`; return `q`. (Uses `math.ceil`.)

2. **`app.py`** (modify) — add `/api/reorder/*` endpoints (all `@require_console_key`, `ok`/`fail`, `from dashboard import reorder as _ro`). No schema-init wiring (no tables).
   - `GET /api/reorder/report` → `reorder_report(plan=None, include_below_par=<query 'below_par' != '0'>)` — the no-plan par-based list.
   - `POST /api/reorder/report` → body `{plan: [{formulation_id, qty}], include_below_par: bool}` → `reorder_report(plan, include_below_par)`. (POST because the plan is a list.)

3. **`static/admin-ingredients.html`** (modify) — add `"reorder"` to the `labels` array + a "Reorder" tab mirroring the Production/Inventory tabs:
   - A plan builder: a search-to-pick formulation picker (reuse the Production tab's `/api/formulations/search` index-array pattern) + a qty input → "Add to plan" appends a `{formulation_id, name, qty}` row to a JS `reorderPlan` array (rendered, removable). The plan is OPTIONAL.
   - An "Include below-par" checkbox (default checked).
   - A "Compute reorder list" button → `POST /api/reorder/report` with `{plan: reorderPlan, include_below_par}` (use the real `api(path,{method:"POST",body:JSON.stringify(...)})`).
   - Render the result grouped by supplier: a section per supplier with a subtotal, each line showing ingredient, on-hand, on-order, par, demand, shortfall, suggested order qty + unit, est cost, and a ⚠ when `unit_warning`. A grand total at the bottom. Read-only (no order buttons).
   - **Add `display:none` initial-state CSS** for the result container if it has an empty/hidden initial state (mirror the recurring pattern). Reuse `api()`/`escapeHtml`/`showTab`. Existing tabs untouched.

4. **`static/console-search-index.json`** (modify) — add `{ "title": "Reorder / Shopping List", "page": "Products", "url": "/admin/ingredients", "keywords": ["reorder","shopping","buy","purchase","demand","bom","par","shortfall","restock"] }`.

5. **Tests** (new):
   - `tests/test_reorder.py` — `bom_demand` explodes a 2-line recipe × qty correctly; `on_order_by_ingredient` sums open-PO ingredient qty minus received and EXCLUDES closed POs + material lines + fully-received lines; `reorder_report` core: (a) below-par with no plan produces a reorder line with correct shortfall = par − on_hand − on_order; (b) a plan adds demand → larger shortfall; (c) an ingredient with on_hand + on_order ≥ par + demand produces NO line; (d) grouping by preferred supplier + subtotal + suggested_qty rounded to MOQ; (e) `_round_up_order` rounds to MOQ and unit_size. Use `tmp_path` + direct schema init (build the few rows by hand, mirroring `tests/test_production.py`).
   - `tests/test_admin_reorder_api.py` — route-level (Pinecone `pytest.skip`): `GET /api/reorder/report` returns groups for a below-par ingredient; `POST /api/reorder/report` with a plan returns a larger shortfall.

## Reuse (don't reinvent)
- `dashboard/inventory.py` `on_hand` + `_connect`/`_default_db_path`; `dashboard/formulations.py` `list_items_for_formulation`; the preferred-source `ORDER BY preferred DESC, price_per_unit` from `ingredient_catalog.list_sources_for_ingredient`.
- 3b `purchase_orders`(status)/`po_items`/`po_receiving` for on-order.
- `dashboard/__init__.py` `require_console_key`/`ok`/`fail`.
- The Production tab's search-to-pick formulation picker (index-array pattern) + the real `api(path,opts)` shape (returns `j.data`, throws) for the console.

## Verification (end-to-end)
1. `pytest tests/test_reorder.py tests/test_admin_reorder_api.py -q` — unit tests green; route tests skip locally on the Pinecone guard.
2. Hand-built temp DB: an ingredient with par 3, on_hand 1, no open PO → `GET /api/reorder/report` lists it with shortfall 2, suggested qty rounded to its source MOQ, grouped under its preferred supplier.
3. Add an open PO receiving-partial for that ingredient → on_order rises → shortfall drops (or the line disappears).
4. `POST` a plan (a formulation that uses the ingredient × qty) → demand raises the shortfall; the line's demand/shortfall reflect it.
5. Console Reorder tab: build a plan, compute, see the grouped shopping list with per-supplier subtotals + grand total; a unit mismatch shows ⚠.

## Build approach
Own writing-plans → subagent-driven-development cycle on branch `sess/59a2725d-p3c3` (stacked on 3c-2, since it reads consumption-affected on-hand + the production recipes), one PR. Tasks: (1) `dashboard/reorder.py` (bom_demand + on_order + reorder_report + helper) + tests — the highest-risk task (the netting formula, the on-order receiving join, MOQ rounding); (2) `/api/reorder/*` endpoints + tests; (3) Reorder console tab + search index. Whole-branch review at the end.

## Out of scope / deferred
- Draft PO creation; unit conversion; saved production plans; material reorder; lead-time scheduling; multi-supplier split-sourcing.
