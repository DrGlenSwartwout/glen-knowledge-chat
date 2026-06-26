# Spec — Console Create Entities (Phase E2)

## Context

E1 (PR #292) made existing FMP-sourced core fields editable with per-field override protection. E2 adds **creation** of brand-new entities in the console — the rest of retiring FileMaker for back-office data, and the missing piece for two pending needs:
- Adding a brand-new ingredient like **HydroCurc/Pharmako** (with its source + price) that isn't in FMP at all.
- The **email-sourcing collector**'s "approve an unmatched quote" — which must create an ingredient + source it can write the price into.

## The clean foundation: `fmp_id = NULL` = console-created = importer-invisible

The importers match FMP rows by `fmp_id`; every table's idempotency index is `... WHERE fmp_id IS NOT NULL`. A row created in the console with **`fmp_id = NULL` is never matched by any re-import** — it is permanently console-owned with no override machinery required. This is the architectural spine of E2: creation just inserts `fmp_id = NULL` rows.

(Trade-off, accepted for v1: if FMP later gains an ingredient with the same name, it imports as a *separate* row — a duplicate to merge by hand. Noted, not solved here.)

## Scope

**In (E2 v1):**
- **Create ingredient** — new `ingredients` row (`fmp_id` NULL): name (required) + form/common_names/par_level/par_level_unit + the curated fields. Optional `canonical_id` link to an existing head (reuse the dedup model) — but default standalone.
- **Create supplier** — new `suppliers` row (`fmp_id` NULL): company (required) + contact fields. Lightweight; needed for new vendors (e.g. Pharmako) not in FMP.
- **Create source** — new `ingredient_sources` row for an ingredient: pick an existing supplier OR a just-created one (or free-text `supplier_name`), + `price_per_unit`/`unit_size`/`unit_type`/`minimum_order`/`minimum_order_unit`/`lead_time_days`/`preferred`.
- **Add recipe item** — new `formulation_items` row on an existing formulation: pick ingredient + dose + dose_unit.
- **Remove recipe item** — delete a `formulation_items` row (a safe leaf row).
- Console UI for each + the `/api/*` endpoints.

**Out (deferred):**
- **Delete ingredient / source / supplier** — referential + re-import-resurrection risk (deleting an FMP row that re-import recreates; deleting a row referenced by recipes/POs/inventory). Later, as careful soft-deactivate (`active` flag), not hard delete. Removing a *recipe item* (leaf) IS in v1; deleting *parents* is not.
- **Create a whole new formulation from scratch** (formulation header + slug + products.json linkage) — E2.5/E3.
- **Editing product-level fields** (`products.json`) — file-regen pipeline, not DB (unchanged from E1).
- Merge/dedup of a console-created ingredient against a later FMP import.

## Data model

**No new tables.** Creation inserts into existing `ingredients` / `suppliers` / `ingredient_sources` / `formulation_items` with `fmp_id = NULL`. One idempotency consideration: the `INSERT OR IGNORE` on partial-unique `fmp_id` doesn't apply to NULLs (NULLs aren't unique-constrained), so plain `INSERT` is correct for created rows. No schema change required; if a `created_source` provenance marker is wanted, set `source_kind`/a flag in `extras` (optional, not required).

## Components & critical files

1. **`dashboard/ingredient_catalog.py`** — new functions (mirror the E1 `set_core_field` validation style):
   - `create_ingredient(fields, db_path=None) -> int` — require non-empty `name`; whitelist the insertable fields (`_ING_CORE` + curated set); `fmp_id` forced NULL; return new id. Reject unknown fields.
   - `create_supplier(fields, db_path=None) -> int` — require non-empty `company`; whitelist supplier fields; `fmp_id` NULL.
   - `create_source(ingredient_id, fields, db_path=None) -> int` — validate the ingredient exists; accept `supplier_id` (must exist if given) and/or `supplier_name`; coerce numerics (`price_per_unit`/`unit_size`/`minimum_order`/`lead_time_days`); `fmp_id` NULL; honor `preferred` (if set, unset others for that ingredient, reusing `set_preferred_source` logic).
   - A shared `_insert_allowed(table, fields, allowed, required, db_path)` helper (mirrors `_update_allowed`): filters to allowed columns, enforces required, INSERTs, returns lastrowid. Numeric coercion via the E1 `_coerce_core` (raise `ValueError` on bad numerics).

2. **`dashboard/formulations.py`** — `add_formulation_item(formulation_id, ingredient_id, dose, dose_unit, db_path=None) -> int` (validate formulation + ingredient exist; coerce `dose`); `remove_formulation_item(item_id, db_path=None)` (DELETE the leaf row; return ok/raise if missing).

3. **`app.py`** — endpoints (all `@require_console_key`, `ok`/`fail`, `ValueError`→`fail(str(e),status=400)` before generic):
   - `POST /api/ingredients` → `create_ingredient` (returns `{id}`).
   - `POST /api/suppliers` → `create_supplier`.
   - `POST /api/ingredients/<int:iid>/sources` → `create_source(iid, body)`.
   - `POST /api/formulations/<int:fid>/items` → `add_formulation_item(fid, ...)`.
   - `DELETE /api/formulation-items/<int:item_id>` → `remove_formulation_item`.

4. **`static/admin-ingredients.html`** — console:
   - Ingredients tab: a **"+ New ingredient"** form (name + the editable fields) → `POST /api/ingredients` → open the new ingredient's detail.
   - Ingredient detail: an **"+ Add source"** sub-form (supplier picker = search existing suppliers + a "new supplier" inline option, price/MOQ/lead-time) → `POST .../sources`.
   - Suppliers tab: a **"+ New supplier"** form.
   - Formulation detail: an **"+ Add ingredient"** row (ingredient search-to-pick + dose + unit) → `POST .../items`; an **✕ remove** on each existing item → `DELETE /api/formulation-items/<id>` (with a confirm).
   Reuse the E1/real `api(path,{method,body:JSON.stringify})` pattern, the search-to-pick index-array pattern (NO `JSON.stringify` in onclick), `escapeHtml`, `toast`.

5. **Tests** —
   - `tests/test_create_entities.py`: `create_ingredient` (fmp_id NULL, name required, unknown field rejected); `create_supplier` (company required); `create_source` (ingredient must exist, numeric coercion, preferred toggling); `add_formulation_item` / `remove_formulation_item` (validation + leaf delete); **the key invariant: a created ingredient (fmp_id NULL) SURVIVES an FMP re-import untouched** (insert created row, re-import the ingredients CSV, assert the created row is unchanged and not duplicated).
   - route tests (Pinecone-skip) for the create/delete endpoints.

## Reuse (don't reinvent)
- E1 `_coerce_core` / allowlist pattern (creation = insert with the same field discipline); `_update_allowed` → mirror as `_insert_allowed`.
- `set_preferred_source` for the `preferred` toggle on a new source.
- The console search-to-pick (formulation/ingredient pickers) + `api()` patterns.
- The `fmp_id IS NULL` importer-invisibility (verified) — created rows need no override flags.

## Verification
1. Unit tests green incl. the created-ingredient-survives-reimport invariant + not-duplicated.
2. Console: create a new ingredient (e.g. **HydroCurc**) → add **Pharmako** as a supplier → add a source ($334/kg, MOQ 25, lead 7–10d) → it appears in the reorder/source views; re-import the ingredients CSV → the created HydroCurc + its source are untouched.
3. Add HydroCurc to a formulation's recipe (dose) → BOM demand for that formulation reflects it; remove it → gone.

## Build approach
spec → plan → SDD, this branch (off the now-merged main), one PR. Tasks: (1) `create_ingredient`/`create_supplier`/`create_source` + `_insert_allowed` + the reimport-survival test; (2) `add/remove_formulation_item`; (3) `/api/*` create/delete endpoints; (4) console create forms. Whole-branch review. **No schema migration** (inserts into existing tables).

## Sequencing & synergy
- Build **after E1** (merged). Unblocks the **email-sourcing collector** (approve-unmatched-quote → `create_ingredient` + `create_source`).
- HydroCurc/Pharmako is the canonical first end-to-end test: create ingredient + supplier + source from the dossier we already ingested.

## Out of scope / deferred (E2.5 / E3)
- Hard/soft delete of ingredients/sources/suppliers (referential-safe deactivate).
- New formulation from scratch (header + products.json linkage).
- Console-vs-FMP duplicate merge tooling.
- Bulk create / CSV paste.
