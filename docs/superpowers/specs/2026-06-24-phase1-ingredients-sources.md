# Phase 1 Spec — Ingredients + Sources → app DB (`chat_log.db`)

## Context

Glen is migrating back-office formulation/ingredient/PO tracking out of FileMaker (FMP) into the deploy-chat app, with the app's SQLite DB (`chat_log.db`) as the **authoritative source** (decisions confirmed 2026-06-24; see `00 System/claude-memory/project_fmp_to_app_migration.md`). Build order is foundation-first: **Ingredients + Sources → Formulations → POs**. This is Phase 1 — the raw-material master + supplier sourcing data that everything else references. It's also the durable fix for the data-hygiene problems surfaced this session: today ingredient data is copied per-product in `products.json` (drift → the cloned-panel bugs we fixed across Macular Wellness / MSM / Nerve). Once ingredients live normalized in the DB, Phase 2 will *generate* the catalog panels from them.

Prior art makes this low-risk: `02 Skills/fmp-loaders/` already designed the entire normalized schema (for a now-dead Supabase mirror). We adapt it to SQLite. The build mirrors the existing `shipping` feature exactly (module + idempotent schema + `/admin` console + `/api` endpoints).

## Scope

**In (confirmed Core + canonicalization):**
- `suppliers` ← FMP `suppliers` (~1,050 rows)
- `ingredients` ← FMP `ingredients` (~2,358 rows), with `canonical_id` clustering applied from the curated `canonical_clusters.csv`
- `ingredient_sources` ← FMP `ingredients_supplier` (~3,974 rows) — the sourcing economics, joined to ingredients + suppliers
- One-time idempotent **importer** from the FMP CSV export
- A **console** (`/admin/ingredients`) to search/view ingredients + their sources/suppliers and edit the curated (non-FMP) fields

**Out (later phases):** materials / material_suppliers / product_suppliers (Phase 3 procurement); formulation→ingredient recipes (Phase 2); POs (Phase 3); generating `products.json` panels from the DB (Phase 2). No customer-facing surface changes in Phase 1.

## Prerequisite (Glen-machine step)

The current `/tmp/fmp-export/newapp/` export only has clients/products/biofield tables — **not** `ingredients`, `suppliers`, `ingredients_supplier`. Re-run the ODBC extractor with FileMaker open + ODBC sharing on to dump those tables:
```
doppler run -p remedy-match -c prd -- ~/.venvs/fmp-ingest/bin/python3 \
  "02 Skills/fmp-odbc-extract.py" --database "Remedy Match" --source newapp \
  --tables ingredients,suppliers,ingredients_supplier
```
The `canonical_clusters.csv` already exists at `02 Skills/fmp-loaders/canonical_clusters.csv`.

## Data model (SQLite, in `chat_log.db`)

Adapted from `fmp-loaders/mapping/00_ingredients_schema.sql` etc. Conventions: `INTEGER PRIMARY KEY AUTOINCREMENT`; timestamps `TEXT DEFAULT (datetime('now'))`; booleans `INTEGER`; arrays/sparse fields as JSON `TEXT`; money as `REAL` (supplier dollars, not customer cents). Each table carries `fmp_id` with a partial unique index (`WHERE fmp_id IS NOT NULL`) — the idempotent re-import key. `PRAGMA foreign_keys=ON` (already set by `_connect`).

```sql
CREATE TABLE IF NOT EXISTS suppliers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fmp_id TEXT, company TEXT NOT NULL,
  address_street TEXT, address_city TEXT, address_province TEXT, address_postal_code TEXT,
  email TEXT, phone_business TEXT, phone_cell TEXT, phone_fax TEXT, url TEXT,
  qb_id TEXT, preferred_contact_type TEXT, active INTEGER,
  notes TEXT,                         -- console-editable (curated)
  extras TEXT,                        -- JSON: fmp sparse fields
  created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_suppliers_fmp ON suppliers(fmp_id) WHERE fmp_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS ingredients (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fmp_id TEXT, name TEXT NOT NULL, form TEXT, status TEXT,   -- FMP-sourced
  common_names TEXT,                  -- JSON array
  canonical_id INTEGER REFERENCES ingredients(id),           -- variant -> head
  extras TEXT,                        -- JSON: dosage/strength/botanical_source/etc.
  -- curated / console-editable, PRESERVED across re-import:
  inci_name TEXT, cas_number TEXT, hygroscopic_rating TEXT, solubility TEXT,
  stability_notes TEXT, spec_notes TEXT, notes TEXT,
  created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ingredients_fmp ON ingredients(fmp_id) WHERE fmp_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS ingredient_sources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fmp_id TEXT,
  ingredient_id INTEGER REFERENCES ingredients(id),
  supplier_id INTEGER REFERENCES suppliers(id),
  supplier_name TEXT, sku TEXT,                              -- FMP-sourced
  price_per_unit REAL, unit_size REAL, unit_type TEXT, shipping_quote REAL,
  -- curated / console-editable, PRESERVED across re-import:
  preferred INTEGER DEFAULT 0, lead_time_days INTEGER,
  minimum_order REAL, minimum_order_unit TEXT, notes TEXT,
  created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ingsrc_fmp ON ingredient_sources(fmp_id) WHERE fmp_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ingsrc_ingredient ON ingredient_sources(ingredient_id);
```

**Curated-vs-FMP split (key design):** the importer only writes FMP-sourced columns; on re-import it `INSERT ... ON CONFLICT(fmp_id) DO UPDATE` refreshing **only** those columns and leaving console-edited curated columns (`preferred`, `lead_time_days`, `minimum_order`, `inci_name`, `notes`, etc.) untouched. This mirrors the Gerald-managed/Phase-B split in the prior loaders and means re-running the import after a fresh FMP export never clobbers Glen's curation.

## Components & critical files

1. **`dashboard/ingredients.py`** (new) — mirrors `dashboard/shipping.py`. Reuse its `_default_db_path()`/`_connect()` (shipping.py:47–57) — factor them into a shared helper or copy the pattern. Exposes:
   - `init_ingredients_schema(cx)` — idempotent CREATE TABLEs above.
   - Read: `list_suppliers`, `get_ingredient(id)`, `search_ingredients(q, limit, offset)`, `list_sources_for_ingredient(id)`.
   - Curated writes: `update_ingredient_curated(id, **fields)`, `update_source_curated(id, **fields)`, `set_preferred_source(source_id)`, `update_supplier(id, **fields)`.
   - All take optional `db_path`, use `with _connect(db_path)`.

2. **`scripts/import_ingredients_from_fmp.py`** (new) — one-time/repeatable importer (pattern of existing `02 Skills/ingest-fmp-products.py`). Reads the 3 FMP CSVs + `canonical_clusters.csv`. Order: suppliers → ingredients (name = first non-empty of name_common/name_compound/name_scientific/name_raw; `active`→status; remaining cols → `extras` JSON) → ingredient_sources (resolve `ingredient_id`/`supplier_id` via `fmp_id` join; map price/purchase_size→unit_size/unit_type/shipping) → canonical pass (validate per `apply_canonical_clusters.py` Stages 1–3: all ids exist, no head-as-member, member appears once; then set `canonical_id = head.id`). Upserts by `fmp_id`, preserves curated columns. `--dry-run` default, `--write`.

3. **`app.py`** (modify) — add `_init_ingredients_tables()` calling `init_ingredients_schema` and invoke at module load (template: shipping.py wiring at app.py:922–928). Add `/admin/ingredients` static-page route + `/api/ingredients/*` JSON endpoints (`@require_console_key`, `ok`/`fail` from `dashboard/__init__.py`): GET search/list, GET ingredient detail (+sources), PATCH ingredient curated, PATCH source curated, POST set-preferred, GET/PATCH suppliers.

4. **`static/admin-ingredients.html`** (new) — mirrors `admin-shipping.html`. Searchable ingredient list (2,358 rows → server-side search + pagination), ingredient detail showing FMP fields read-only + editable curated fields + its sources (supplier, price, unit, sku) with a "preferred" toggle; a suppliers tab (view/edit contact). Don't restyle; match existing console look.

5. **`static/console-search-index.json`** (modify) — register `{ "title": "Ingredients & Sources", "page": "Products", "url": "/admin/ingredients", "keywords": ["ingredient","ingredients","sources","supplier","raw","cost","sourcing"] }`.

6. **Tests** (new) — `tests/test_ingredients.py` (schema + CRUD + curated-preserve-on-reimport, `tmp_path` fixture like test_shipping.py:87–120), `tests/test_import_ingredients.py` (importer parsing + canonical pass + idempotent re-import preserves curated), `tests/test_admin_ingredients_api.py` (endpoints; `_load_app()` Pinecone-skip pattern from test_bos_routes.py:8–15).

## Reuse (don't reinvent)
- `dashboard/shipping.py` `_connect`/`_default_db_path`, idempotent `init_*_schema`, CRUD-with-`db_path` shape.
- `dashboard/__init__.py` `require_console_key`, `ok()`, `fail()`.
- `app.py:922–928` `_init_shipping_tables()` as the schema-init-at-load template.
- `02 Skills/fmp-loaders/` SQL as the field-mapping reference; `apply_canonical_clusters.py` for the validation logic; `canonical_clusters.csv` as input.
- `02 Skills/fmp-odbc-extract.py` for the re-export; `02 Skills/ingest-fmp-products.py` as the importer skeleton.

## Verification (end-to-end)
1. Re-export the 3 FMP tables (prereq command above); confirm CSVs land in `/tmp/fmp-export/newapp/`.
2. `pytest tests/test_ingredients.py tests/test_import_ingredients.py tests/test_admin_ingredients_api.py -q` — all green.
3. Import dry-run → review counts (~1,050 suppliers / ~2,358 ingredients / ~3,974 sources); `--write` into a temp `chat_log.db`.
4. Assert: row counts match; `canonical_id` set for known clustered members (e.g. the 24 CBD variants point to head); a spot ingredient (e.g. R-Lipoic Acid) shows its sources with supplier + price.
5. Edit a curated field in the console (mark a source `preferred`, set a `lead_time_days`); re-run the importer `--write`; confirm the curated edit survived and FMP columns refreshed.
6. Load `/admin/ingredients?key=...`, search an ingredient, open detail, verify sources + suppliers render.

## Build approach
This spec becomes its own writing-plans → subagent-driven-development cycle (same as the shipping/packer and catalog work), on a fresh branch off `main`, shipped as one PR. Each component (schema/module, importer, console, tests) is a task; the importer + canonical pass is the highest-risk task and gets adversarial review.

## Out of scope / deferred
- Materials, material_suppliers, product_suppliers (Phase 3).
- Formulation→ingredient recipes and generating `products.json` panels from the DB (Phase 2 — will retire `scripts/refresh_ingredients_from_fmp.py`).
- Purchase orders + BOM demand (Phase 3).
