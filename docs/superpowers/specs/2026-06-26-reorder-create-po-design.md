# Reorder → Draft PO (Sub-project C1) — Design Spec

**Date:** 2026-06-26
**Status:** Design approved. First slice of sub-project **C** (console inline "act-where-shown"
actions). C2 (money actions) and C3 (cross-links) are separate cycles, out of scope here.

## Goal

Let an operator turn the reorder report's shortfall lines into a **draft purchase order** with one
click, per supplier group — closing the audit's single biggest procurement gap (the reorder report
is read-only today). C1 **creates the draft only**; reviewing/editing happens in the existing
Purchase Orders tab, and actually ordering/sending a PO (vendor PO #, supplier email) is a future
step, not C1.

## Current state (reuse-first)

- **`dashboard/reorder.py` `reorder_report(plan, include_below_par, db_path)`** returns
  `{groups:[{supplier_id:int|None, supplier:str, subtotal:float, lines:[{ingredient_id, ingredient,
  suggested_qty, unit, price_per_unit, unit_size, packs, est_cost, …}]}], totals, plan_echo}`. The
  `supplier_id` is already at group level (from the preferred `ingredient_sources` row). Lines with no
  source land in the `supplier_id: None` group ("— no preferred source —").
- **`dashboard/purchase_orders.py`** is read + curated-notes only — **there is NO forward-creation
  function** (the FMP migration imported PO history + deferred forward creation). Tables:
  `purchase_orders(id, fmp_id, supplier_id, supplier_name, vendor_po_no, po_date, status, tax,
  shipping_amount, …, extras, notes, created_at, updated_at)`; `po_items(id, fmp_id, po_id, item_kind,
  item_label, ingredient_id, material_id, …, qty, qty_unit, qty_left, cost, extras, notes, …)`.
- **BOS action layer:** `dashboard/actions.py` `Action(key, module, title, description, risk_tier,
  permission, executor, confirm_summary, reversible)` + `register_action`; `dispatch.py
  dispatch_action(cx, key, params, actor, …)`; `rbac.py` policy (LOW_WRITE → AUTO for OWNER/OPS).
  HTTP: `POST /api/action/<key>` (`bos_action` → `_bos_actor` → `dispatch_action`). Action modules are
  registered in `app.py` (~line 24095+, e.g. `from dashboard import reviews_actions as _ra`). Example:
  `reviews.approve` = `Action(key="reviews.approve", module="reviews", risk_tier=LOW_WRITE,
  permission=(OWNER, OPS), executor=_exec_approve)`.
- **UI:** `static/admin-ingredients.html` `renderReorderReport(el, rep)` renders the grouped shopping
  list (a per-supplier `<h4>` header + an 8-column table); it's the single render path used by both
  the manual (`roCompute`) and velocity (`loadVelocity`) reports. The Purchase Orders tab
  (`showTab('po')`) is search-only with `openPo(id)` to open a PO's detail. Tabs are switched by
  `showTab('<id>')` (no URL-param support today).
- `static/console-products.html` Backorders tab is **product-level** (`backorder_rollup` →
  `{slug, name, units_backordered, order_count}`) — incompatible with the reorder report's
  ingredient/formulation plan, so a true pre-filled bridge isn't possible.

## Design

### Component 1 — `purchase_orders.create_draft_po(cx, supplier_id, supplier_name, lines)`

New function. Inserts one draft PO + its line items, returns the new id.
- INSERT `purchase_orders`: `status='draft'`, `supplier_id`, `supplier_name`, `po_date=date('now')`,
  `vendor_po_no = 'DRAFT-' + YYYYMMDD + '-' + supplier_id` (findable via the existing PO search; not
  required unique), `fmp_id=NULL`. Capture the new `po_id`.
- For each line in `lines`, INSERT `po_items`: `po_id`, `item_kind='ingredient'`,
  `item_label=<ingredient name>`, `ingredient_id`, `qty=<suggested_qty>`, `qty_unit=<unit>`,
  `cost=<price_per_unit>` (the pack price; **NULL when the line has no price** — included, not dropped),
  `extras=json({unit_size, packs, est_cost})`.
- Returns `{"po_id": int, "line_count": int}`. Commits.
- Add to the module's read-side patterns; no schema change (tables already exist via
  `init_purchase_orders_schema`).

### Component 2 — `dashboard/reorder_actions.py` (new) — `reorder.create_po`

Register a BOS action mirroring `reviews_actions.py`:
```
Action(key="reorder.create_po", module="reorder", title="Create draft PO",
       description="Create a draft purchase order from a reorder-report supplier group.",
       risk_tier=LOW_WRITE, permission=(OWNER, OPS), executor=_exec_create_po)
```
Executor `_exec_create_po(params, ctx)`:
- `supplier_id = params.get("supplier_id")` (int; reject/return error if None — the no-source group
  can't make a PO), `supplier_name = params.get("supplier_name") or ""`.
- **Sanitize** `params.get("lines")`: keep only `{ingredient_id, ingredient, suggested_qty, unit,
  price_per_unit, unit_size, packs, est_cost}` per line, coercing numbers; drop lines without an
  `ingredient_id` or `suggested_qty`. (Internal, owner-gated, draft-and-review → client-sent lines are
  acceptable; sanitizing guards against malformed input.)
- Call `purchase_orders.create_draft_po(ctx["cx"], supplier_id, supplier_name, sanitized)`; return its
  result. `register()` is idempotent (guard with `get_action("reorder.create_po")`).
- Wire in `app.py`: `from dashboard import reorder_actions as _roa` + `_roa.register()` in the action-
  registration block (~line 24129). `create_draft_po` needs the same `cx` the dispatcher passes (the
  action receives `ctx["cx"]`), so the function takes a connection, not a db_path.

### Component 3 — UI: "Create draft PO" button (`static/admin-ingredients.html`)

In `renderReorderReport(el, rep)`:
- **Stash the rendered report** in a module-scoped var (e.g. `_lastReorderRep = rep`) so the button can
  reference a group by **index** — do **NOT** `JSON.stringify` the lines into the `onclick` attribute
  (that pattern has repeatedly broken in this codebase on quotes/apostrophes). The button passes only
  the integer group index.
- In each supplier-group header: if `g.supplier_id != null`, append
  `<button class="..." onclick="createDraftPO(<groupIndex>)">Create draft PO</button>`. For the
  `supplier_id == null` group, append a muted note instead: "Assign a preferred source to order these."
- `createDraftPO(idx)`: read `g = _lastReorderRep.groups[idx]`; `confirm("Create draft PO for " +
  g.supplier + " — " + g.lines.length + " lines, ~$" + g.subtotal + "?")`; on OK, `POST
  /api/action/reorder.create_po` (via the page's existing `api()` helper / `X-Console-Key`) with
  `{supplier_id: g.supplier_id, supplier_name: g.supplier, lines: g.lines}`. On `{po_id}` success:
  `showTab('po')` then `openPo(po_id)` so the new draft opens immediately; on error, a toast/alert.
- **Add `?tab=` support** to the page init (read `new URLSearchParams(location.search).get('tab')` on
  load → `showTab(tab)` if it's a valid tab id) so the Backorders link (Component 4) can land on the
  Reorder tab.

### Component 4 — Backorders shortcut (`static/console-products.html`)

In the Backorders tab, add a simple link: `<a href="/admin/ingredients?tab=reorder&key=<key>">Open
reorder report →</a>` (carry the console key the page already holds). No data bridge — product-level
backorders don't map to the ingredient/formulation reorder plan.

## Out of scope

- Ordering/sending a PO (vendor PO #, supplier email, marking ordered) — a future step.
- Editing PO line items in the UI beyond what the PO tab already supports (curated `notes`).
- C2 (money actions), C3 (cross-links). A true Backorders→reorder data bridge.

## Dependencies

- `dashboard/reorder.py` (report shape), `dashboard/purchase_orders.py` (tables + read helpers),
  the BOS action layer (`actions`/`dispatch`/`rbac`), and Rae's OWNER token (sub-project A) — Rae is
  `OPS`... note: Rae's scoped token resolves to OWNER via `actor_for_scope` (she's a co-owner), and the
  action permits `(OWNER, OPS)`, so both Glen and Rae can create draft POs.

## Testing (run via [reference_deploy_chat_local_tests])

- **`create_draft_po`** (unit, seed suppliers + ingredients): creating from 2 lines (one priced, one
  with `price_per_unit=None`) inserts 1 `purchase_orders` row (`status='draft'`, `vendor_po_no` starts
  `DRAFT-`, correct `supplier_id`) + 2 `po_items` (`qty`/`qty_unit`/`ingredient_id` correct;
  `cost=price_per_unit` for the priced line and `cost IS NULL` for the no-price line; `extras` carries
  `unit_size`/`packs`/`est_cost`); returns `{po_id, line_count:2}`.
- **`reorder.create_po` action** (dispatch): with an OWNER actor, dispatching `reorder.create_po` with
  `{supplier_id, supplier_name, lines:[…]}` returns the executor result (a `po_id`) and the PO exists;
  a `supplier_id=None` payload returns an error result (no PO created); the action's `permission` is
  `(OWNER, OPS)` and `risk_tier` is `LOW_WRITE`.
- **Render-verify (headless, per the render-verify lesson) — mocked reorder report:** with
  `/api/reorder/report` mocked to a 2-group report (one with `supplier_id`, one `null`), the Reorder
  tab shows a **Create draft PO** button on the priced group and the **no-source note** on the null
  group; clicking the button (with `/api/action/reorder.create_po` mocked to `{ok:true, po_id:7}`)
  switches to the PO tab and calls `openPo(7)`; **zero JS console/page errors** and **no JSON-in-onclick
  breakage** (a supplier name with an apostrophe still works). `/admin/ingredients?tab=reorder` opens
  the Reorder tab. The Backorders tab shows the "Open reorder report →" link.

## Rollout

Additive: one new `purchase_orders` function, one new action module + its registration, a button +
`createDraftPO` + `?tab=` support in admin-ingredients, and a link in console-products. No schema
change, no feature flag. Console-key / OWNER-token gated; the action is RBAC-gated (LOW_WRITE).
