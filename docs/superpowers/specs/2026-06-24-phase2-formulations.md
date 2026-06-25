# Phase 2 Spec — Formulations (recipes) → app DB

**Date:** 2026-06-24
**Status:** Decisions confirmed 2026-06-24 (Glen). Ready to turn into an implementation plan.
**Repo:** deploy-chat. Builds on Phase 1 (PR #273, `dashboard/ingredient_catalog.py` + ingredients/suppliers/ingredient_sources tables).

## Context
Phase 2 of the FileMaker → app-DB migration. Phase 1 put the raw-material master (ingredients/suppliers/sources) into `chat_log.db` as the authoritative source. Phase 2 adds the **recipes**: which ingredients (and doses) make up each formulation, referencing Phase-1 ingredients by id. This is also the durable end of the stale/cloned ingredient-panel bug class we fixed repeatedly: once recipes live in the DB, the customer-facing `products.json` ingredient panels are **generated** from them (one source of truth), retiring `scripts/refresh_ingredients_from_fmp.py`.

Prior art: `fmp-loaders/mapping/07_formulations.sql` + `08_formulation_ingredients.sql` already model this; adapt to SQLite, same patterns as Phase 1.

## Source data (FMP, already exportable via the AppleScript extractor)
- **Formulations** = FMP `products` rows where `type = 'Functional Formulation'` (**181 rows**).
- **Recipe lines** = FMP `products_items` (`id_fk_product` → product; `id_fk_raw` → ingredient; `qty`, `unit_measurement`, `zc_mg`, `zc_raw_display`). Of 1,742 rows, **1,684 belong to FF products; 1,274 carry an `id_fk_raw` ingredient link** (the rest are material/packaging or blank-named lines — same incompleteness we saw in the refresher; those route to review, not silent loss).
- Re-export adds `products` + `products_items` to the extractor `--tables` list (note: `products.csv`/`products_items.csv` already present from prior exports).

## Scope

**In:**
- `formulations` table ← FMP FF products (fmp_id, name, status, extras; link to storefront product — see Decision A).
- `formulation_items` table ← FMP `products_items` (formulation_id, ingredient_id → Phase-1 `ingredients`, dose, unit, raw_text). Lines with no resolvable `id_fk_raw` → flagged, not dropped.
- Idempotent importer `scripts/import_formulations_from_fmp.py` (by fmp_id; curated-vs-FMP split like Phase 1).
- `/admin/formulations` console (or extend `/admin/ingredients`): view a formulation's recipe, see each ingredient's Phase-1 sourcing, edit curated recipe notes.
- **Panel generator** (Decision B): one-way DB → `products.json` `ingredients` `[{name,dose}]`, retiring `refresh_ingredients_from_fmp.py`.

**Out (Phase 3):** purchase orders, BOM demand, materials/packaging. No new customer-facing UI beyond the generated panels.

### Decision A — recipe↔storefront-product link (CONFIRMED: add `fmp_id` to products.json)
FMP recipes key off the FMP product id; `products.json` uses slugs with no FMP id. **Confirmed:** a one-time matcher stores `fmp_id` on each storefront product (reusing the Phase-1/bottle-type fuzzy/alias matcher), giving a stable exact link reused by Phase 2 + 3. Unmatched products → review list (manual, like bottle types). *Alternative:* name-match at generate time (simpler, but fragile — the Synergy/Syntropy near-miss class). Recommend the stable id.

### Decision B — generate panels now (CONFIRMED: yes, retire the refresher)
**Confirmed:** Phase 2 includes the one-way generator and retires `refresh_ingredients_from_fmp.py`, making the DB the sole source of the customer ingredient panels. Aligns with the confirmed "DB = authoritative" decision. *Alternative:* defer the generator, keep the refresher during a transition.

## Data model (SQLite, `chat_log.db`)
Same conventions as Phase 1 (fmp_id partial unique index, curated-vs-FMP split, JSON extras, INTEGER bools, REAL numbers).

```sql
CREATE TABLE IF NOT EXISTS formulations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fmp_id TEXT, name TEXT NOT NULL, status TEXT,
  product_slug TEXT,              -- link to products.json (Decision A); nullable
  extras TEXT,
  notes TEXT,                     -- curated
  created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_formulations_fmp ON formulations(fmp_id) WHERE fmp_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS formulation_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fmp_id TEXT,
  formulation_id INTEGER REFERENCES formulations(id),
  ingredient_id INTEGER REFERENCES ingredients(id),   -- Phase 1
  ingredient_name TEXT,           -- denormalized for panel generation + unresolved rows
  dose REAL, dose_unit TEXT, raw_text TEXT,
  extras TEXT,
  notes TEXT,                     -- curated
  created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_formitems_fmp ON formulation_items(fmp_id) WHERE fmp_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_formitems_form ON formulation_items(formulation_id);
```

## Importer (`scripts/import_formulations_from_fmp.py`)
- Reuse Phase-1 helpers (`_active`/`_num`/`_clean`/`_extras`/`_upsert`) — factor them into a shared `dashboard/fmp_import_util.py` to avoid duplication.
- `import_formulations`: FMP products (type=FF) → formulations (name, status, extras; product_slug via Decision-A matcher). Curated `notes` preserved.
- `import_formulation_items`: products_items belonging to FF products → formulation_items; resolve `ingredient_id` via `id_fk_raw`→`ingredients.fmp_id`; dose from `zc_mg`/`qty`+`unit_measurement`; keep `ingredient_name`/`raw_text`. Unresolved `id_fk_raw` → row kept with ingredient_id NULL + flagged in the run report.
- Idempotent by fmp_id; curated-preserving; dry-run/`--write`.

## Panel generator (`scripts/generate_panels_from_db.py`) — Decision B
- For each formulation with a `product_slug`, build `ingredients = [{name, dose}]` from its `formulation_items` (name from the linked Phase-1 ingredient, fallback to `ingredient_name`; dose from dose+unit). Apply a **completeness guard** (like the refresher: a formulation with unresolved/dosed-but-unnamed lines → review, don't overwrite a good panel).
- `--write` patches `products.json` panels for matched, complete formulations only; sets `ingredients_source = "db-formulations-<date>"`. Dry-run default; review list for the rest.
- On adoption, delete `scripts/refresh_ingredients_from_fmp.py` (superseded).

## Console
Extend the Phase-1 console (or a sibling page): a formulation list/search; detail shows the recipe (ingredient, dose, unit) with each ingredient's preferred source/cost from Phase 1; editable curated `notes` on formulation + items. Same `@require_console_key` / `ok`/`fail` / `?key=` patterns.

## Tests
- `tests/test_formulations.py` (schema + reads + curated).
- `tests/test_import_formulations.py` (FMP join to Phase-1 ingredients; unresolved-line handling; idempotent curated-preserve).
- `tests/test_generate_panels.py` (panel build from formulation_items; completeness guard holds back incomplete; never overwrites unmatched).
- Route tests (Pinecone-skip pattern).

## Verification
Re-export FMP products/products_items → importer `--write` into a temp DB → assert 181 formulations, ~1,274 resolved recipe lines, unresolved flagged. Generator dry-run → review count; spot-check a known formula's generated panel matches its FMP recipe. Curated edit survives re-import.

## Non-goals / deferred
- Materials/packaging lines (the non-`id_fk_raw` products_items rows) — Phase 3.
- POs + BOM demand — Phase 3.
- Backfilling `fmp_id` onto every storefront product is bounded to FF products with recipes; non-FF catalog items keep their current panels until later.
