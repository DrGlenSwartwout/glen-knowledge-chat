# Spec — Console Core-Field Editing (Phase E1) with Override Protection

## Context

The FMP→app-DB migration made `chat_log.db` the authoritative store, but the console (`/admin/ingredients`) only edits the **curated overlay** fields — it's read-only for FMP-sourced core data (names, recipe doses, prices, par levels). To retire FileMaker for day-to-day work, those core fields must become editable in the console, **without** a future FMP re-import clobbering the edits.

This is **Phase E1**: edit *existing* core fields, with **per-field override protection**. Creating new ingredients/sources/recipe-lines/formulations is **Phase E2** (deferred). Product-level fields (customer name/retail price) live in `products.json` (read-only at runtime on Render) and are out of scope — they change via the file-regen pipeline, not live DB edits.

Decisions confirmed by Glen: **Option A (edit existing core fields)** + **override-protection** (not a hard cutover).

## The override mechanism (architectural heart)

Today's curated-vs-FMP split is **column-level**: a column is *either* always-FMP (refreshed on import) *or* always-curated (never touched). E1 generalizes this to **per-row, per-field**: when the console edits a core FMP field on a row, that field is recorded in the row's `overrides`, and the importer stops refreshing *that field on that row* — while still refreshing it on every untouched row.

- Each editable table gains an `overrides TEXT` column — a JSON array of the field names that have been console-edited (and are now console-owned).
- The shared `_upsert` (in `scripts/import_ingredients_from_fmp.py`) is made override-aware: its currently-**unused** `conflict_update_cols` parameter becomes the list of columns to UPDATE on re-import. Importers compute, per row, `update_cols = fmp_cols − overrides_for_that_row` and pass it. New rows still INSERT all FMP values (no overrides yet); existing rows refresh only non-overridden columns.
- A console edit calls `set_core_field(...)` → writes the value AND adds the field to `overrides` (idempotent set-add).
- A **"revert to FMP"** action calls `revert_core_field(...)` → removes the field from `overrides` (the value is left as-is; the next FMP import, if any, will overwrite it). This *unlocks* the field; it does not restore a prior value (we don't keep a shadow copy — YAGNI).

This is a safe transition, not a one-way switch: a final/occasional FMP re-import still works for everything you haven't touched, and your edits are protected and visibly flagged.

## Scope

**In (E1):**
- `overrides` column on `ingredients`, `ingredient_sources`, `formulation_items`.
- Override-aware `_upsert` (uses `conflict_update_cols`); the **ingredients** and **formulations** importers preload an overrides map and pass per-row `update_cols`.
- **Promote `par_level` / `par_level_unit`** from the `ingredients.extras` JSON to real columns (reorder-critical; cleanest as first-class editable+overridable columns). Backfill from extras; repoint readers.
- Editable core fields:
  - `ingredients`: `name`, `form`, `par_level`, `par_level_unit`
  - `ingredient_sources`: `price_per_unit`, `unit_size`, `unit_type`
  - `formulation_items`: `dose`, `dose_unit`
- Module functions `set_core_field` / `revert_core_field` with a per-table **allowlist** of editable core fields (mirrors the existing `_update_allowed` curated pattern).
- `/api/*` endpoints for core-field PATCH + revert.
- Console UI: turn those read-only fields into editable inputs in the Ingredients, Suppliers/Sources, and Formulations tabs, each with an **"overridden ⟳ revert"** indicator.
- Tests proving: edit writes value + override; re-import skips overridden fields on that row but refreshes others and other rows; revert unlocks; par_level column + backfill + readers.

**Out (deferred / E2 / separate):**
- Create/delete: new ingredients, sources, formulations; **add/remove recipe items** (E1 edits doses on *existing* items only).
- Product-level fields (`products.json` name/price) — file-regen pipeline, not DB.
- Editing core fields on materials / purchase_orders / production (no day-to-day need; their importers keep the as-is `_upsert` path).
- A shadow/history of prior FMP values; full audit log.

## Data model changes (SQLite, `chat_log.db`)

```sql
-- per-row override tracking (JSON array of console-owned field names)
ALTER TABLE ingredients         ADD COLUMN overrides TEXT;   -- e.g. '["par_level","name"]'
ALTER TABLE ingredient_sources  ADD COLUMN overrides TEXT;
ALTER TABLE formulation_items   ADD COLUMN overrides TEXT;

-- promote par to first-class columns (was ingredients.extras.par_level)
ALTER TABLE ingredients ADD COLUMN par_level REAL;
ALTER TABLE ingredients ADD COLUMN par_level_unit TEXT;
```

All `ADD COLUMN` are idempotent-guarded in the schema-init (check `PRAGMA table_info` before adding, matching how the repo evolves schemas), so deploy is safe and re-runnable. `inventory_starting` stays in `extras` (seed-only, never edited).

**Backfill (one-time, in schema-init or a tiny migration):**
```sql
UPDATE ingredients
   SET par_level      = CAST(json_extract(extras,'$.par_level')      AS REAL),
       par_level_unit = json_extract(extras,'$.par_level_unit')
 WHERE par_level IS NULL AND json_extract(extras,'$.par_level') IS NOT NULL;
```

## Components & critical files

1. **`scripts/import_ingredients_from_fmp.py`** — make `_upsert` override-aware:
   - Build a `col→value` map from `["fmp_id"]+fmp_cols` / `values`.
   - INSERT OR IGNORE all FMP cols (unchanged — new rows have no overrides).
   - UPDATE only `conflict_update_cols` (currently the body ignores this param and uses `fmp_cols`; switch it), binding values via the map. **All existing callers already pass `fmp_cols` as `conflict_update_cols`, so their behavior is unchanged.**
   - Add `par_level`/`par_level_unit` to the ingredients importer's `fmp_cols` (mapped from `r.get("par_level")`/`r.get("par_level_unit")`); remove them from the `extras` blob (add to the `mapped` set).
   - In `import_ingredients` and `import_sources`: preload `overrides_map = {fmp_id: set(json.loads(overrides or '[]'))}` for the table, and per row pass `update_cols = [c for c in fmp_cols if c not in overrides_map.get(fid, ())]`.

2. **`scripts/import_formulations_from_fmp.py`** — same override-aware preload + per-row `update_cols` for `formulation_items` (protect overridden `dose`/`dose_unit`). The other importers (materials, POs, production) are untouched (no editable core fields in E1; they keep passing `fmp_cols`).

3. **`dashboard/ingredient_catalog.py`** + **`dashboard/formulations.py`** — schema-init adds the `overrides` columns + `par_level` columns + backfill (idempotent). New functions:
   - `set_core_field(row_id, field, value, db_path=None)` per table (ingredient / source / formulation-item), gated by an allowlist (`_ING_CORE = {"name","form","par_level","par_level_unit"}`, `_SRC_CORE = {"price_per_unit","unit_size","unit_type"}`, `_ITEM_CORE = {"dose","dose_unit"}`): writes the field (with numeric coercion for numeric fields) AND adds `field` to that row's `overrides` JSON. Returns the updated row.
   - `revert_core_field(row_id, field, db_path=None)`: removes `field` from `overrides` (value untouched). Returns the updated row + a note that the next FMP import will refresh it.
   - Reads that currently `SELECT *` already return `overrides`, `par_level`, `par_level_unit` once the columns exist — no change. (Reorder/inventory readers below DO change.)

4. **`dashboard/reorder.py`** + **`dashboard/inventory.py`** — repoint par readers from `json_extract(extras,'$.par_level')` to the new `par_level` / `par_level_unit` columns (reorder.py:92,100,101; inventory.py:52-53,82-83 and the `inventory_levels`/`get_inventory`/seed queries). The inventory **seed** reads `par_level_unit` from extras (line 148) — switch to the column. `inventory_starting` stays in extras (line 147,150). This is the blast radius of the promotion; it's mechanical and covered by existing inventory/reorder tests plus new assertions.

5. **`app.py`** — endpoints (all `@require_console_key`, `ok`/`fail`):
   - `PATCH /api/ingredients/<int:id>/core` — body `{field, value}` → `set_core_field`; 400 on a non-allowlisted field.
   - `PATCH /api/sources/<int:id>/core` and `PATCH /api/formulation-items/<int:id>/core` — same shape.
   - `POST /api/<entity>/<int:id>/revert` — body `{field}` → `revert_core_field`.
   (Or one generic `/api/core-edit` taking `{entity, id, field, value}` — implementer's call; the brief will pick one to keep the surface small.)

6. **`static/admin-ingredients.html`** — in the Ingredients detail, the per-ingredient **sources** rows, and the Formulations **recipe items**: render the in-scope fields as editable inputs (save on blur/Enter → the core PATCH). Show an **"overridden"** badge (amber) next to any field whose name is in the row's `overrides`, with a small **⟳ revert** control (→ the revert endpoint, then refresh). Read-only fields stay read-only. Reuse the real `api(path,{method:"PATCH",body:JSON.stringify(...)})` pattern; `escapeHtml`; no `JSON.stringify` inside any onclick (index/dataset pattern).

7. **Tests** (new + extended):
   - `tests/test_core_edit.py` — `set_core_field` writes value + adds to `overrides`; allowlist rejects an unknown field; `revert_core_field` removes from `overrides`; numeric coercion.
   - `tests/test_import_override.py` — seed a row, mark `price_per_unit` overridden, re-import with a *different* FMP price → overridden field unchanged, a non-overridden field on the same row refreshed, and the same field on a *different* (non-overridden) row refreshed. Proves per-row, per-field protection.
   - extend `tests/test_inventory.py` / `tests/test_reorder.py` — par_level read from the column (+ the extras→column backfill path).
   - route tests (Pinecone-skip) for the core PATCH + revert endpoints.

## Reuse (don't reinvent)
- `_upsert` + the `_update_allowed` curated-write helper as the templates (core-edit is curated-write + an `overrides` set-add).
- The schema-init `ADD COLUMN`-if-absent idiom already used in the repo.
- Console editing patterns from the existing curated-field editors (notes/preferred) in `admin-ingredients.html`.

## Verification (end-to-end)
1. Unit tests green (core-edit, override-on-reimport, par column/backfill, routes skip on Pinecone).
2. On a temp DB: edit an ingredient's `par_level` in the console → reorder uses the new value; mark it overridden; re-import the ingredients CSV with a changed par → the overridden par survives, a non-overridden ingredient's par refreshes.
3. Edit a source `price_per_unit` → reorder cost reflects it; revert → field unlocks (next import would refresh).
4. Edit a recipe item `dose` → the BOM demand for that formulation changes accordingly.
5. Existing inventory/reorder behavior unchanged for non-edited data.

## Build approach
Spec → writing-plans → subagent-driven-development on a fresh branch off `main`, one PR. Tasks (foundation-first): (1) override-aware `_upsert` + par_level promotion/backfill + reader repoint + tests [highest-risk: the import-protection invariant and the par blast radius]; (2) `set_core_field`/`revert_core_field` + allowlists + tests; (3) `/api/*` core-edit + revert endpoints + route tests; (4) console editable fields + overridden/revert UI. Whole-branch review at the end. **Activation after merge: edit core fields in the console; a final FMP re-import (if ever run) refreshes only non-overridden fields.**

## Out of scope / deferred (E2 and beyond)
- Create/delete rows; add/remove recipe ingredients; new formulations/products.
- Product-level (`products.json`) field editing.
- Core editing for materials/POs/production.
- Prior-value shadow / audit history.
- Bulk edit; CSV round-trip export.
