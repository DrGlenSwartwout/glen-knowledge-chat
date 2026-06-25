# Phase 3c-2 Spec — Production Runs + Consumption

## Context

Phase 3c gives the back-office reorder intelligence (`chat_log.db` authoritative; see `00 System/claude-memory/project_fmp_to_app_migration.md`). 3c-1 built the **inventory ledger** (`inventory_txns`, signed `SUM` = on-hand) and wired the *inflow* side: `baseline` (from `ingredients.extras.inventory_starting`) + `receipt` (from `po_receiving`). The `consumption` txn_type exists in the schema but **nothing writes it yet**.

3c-2 wires the *outflow* side: production runs consume ingredients, so each run posts negative `consumption` entries to the ledger. This completes Glen's stated model:

```
on_hand = baseline + purchases (receipts) − production (consumption) ± recounts
```

It imports the historical FMP production data (179 runs / 3,325 line items) and adds a console to view runs and log new ones going forward (a new run posts consumption as it's recorded — the "update as we do production runs" path).

### The baseline-date approximation (important, handled as a runtime knob)

The baseline (`inventory_starting`) is a count with **no date** in FMP. Glen's model treats it as the anchor and applies purchases/production as flows. Strictly, a production run that predates the baseline count is already reflected in the baseline, so posting it again over-subtracts. Glen's accepted approach is *approximate-then-recount*. To avoid hard-coding a choice that could make on-hand read alarmingly low, **consumption posting is controlled at activation**:
- `post all` (default) — full model: every historical run posts consumption.
- `post from <date>` — only runs on/after a cutoff post consumption (the rest import as records).
- `record only` — import runs as records; post no consumption (the going-forward-only stance).

The same toggle lives on the server-side import endpoint. Manual recounts (3c-1 `add_adjustment`) true up any residual drift. (Symmetry note: 3c-1 seeds *all* receipts; if Glen later wants a receipt cutoff too, that's a one-line filter follow-up — out of scope here.)

## Scope

**In:**
- `production_runs` + `production_run_items` tables (mirror FMP `production` / `production_items`; adapt the `02 Skills/fmp-loaders/mapping/13_*`+`14_*` Postgres mappings to SQLite).
- Importer from the FMP CSVs (idempotent by `fmp_id`), resolving runs→formulations and items→ingredients/materials.
- **Consumption posting** into `inventory_txns` (negative, idempotent by `source_ref`), honoring the `post all | post from date | record only` mode.
- `dashboard/production.py` module (reads + curated `notes` writes + `log_run` for new console-entered runs + `post_consumption`).
- Console: a "Production" tab in `/admin/ingredients` (run list, run detail with line items + consumption status, a "Log a production run" form that can pre-fill lines from the formulation's recipe and posts consumption on save).
- `/api/production/*` endpoints + a server-side `POST /api/production/import` (upload the 2 CSVs + the consumption mode) mirroring the Phase-3b/3c-1 import endpoints.

**Out (deferred):**
- **Material** consumption into a ledger (production_run_items records material lines as *data*, but only `ingredient`-typed lines post to `inventory_txns` — materials inventory is not tracked, consistent with 3c-1 receipts skipping material-only lines).
- BOM demand / reorder list (3c-3).
- Editing/voiding a historical run's already-posted consumption (a void/reversal flow is later; for now a recount corrects).
- Recipe-scaling math beyond a simple editable pre-fill (see below).

## Prerequisite (Glen-machine step, needed only to POPULATE — not to build)

Re-export the two FMP production tables with FileMaker open + the AppleScript extractor (the ODBC path fails on Apple Silicon):
```
~/.venvs/fmp-ingest/bin/python3 "02 Skills/fmp-applescript-extract.py" \
  --source newapp --tables production,production_items
```
→ `/tmp/fmp-export/newapp/production.csv` (~179 rows) + `production_items.csv` (~3,325 rows). Build + tests use synthetic fixtures; the real import is an activation step.

## Data model (SQLite, in `chat_log.db`)

Conventions match 3b/3c-1: `INTEGER PRIMARY KEY AUTOINCREMENT`; `REAL` quantities; `TEXT DEFAULT (datetime('now'))`; `fmp_id` partial-unique for idempotent re-import; curated `notes` preserved across re-import.

```sql
CREATE TABLE IF NOT EXISTS production_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fmp_id TEXT,
  formulation_id INTEGER REFERENCES formulations(id),
  batch_number TEXT,
  run_date TEXT,                  -- 'YYYY-MM-DD' (FMP production_date)
  quantity_units REAL,            -- FMP qty (units/bottles produced)
  status TEXT,                    -- 'completed' for imported historical runs
  source_kind TEXT,               -- 'fmp' | 'manual'
  extras TEXT,
  notes TEXT,                     -- curated
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_prodrun_fmp ON production_runs(fmp_id) WHERE fmp_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_prodrun_form ON production_runs(formulation_id);

CREATE TABLE IF NOT EXISTS production_run_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fmp_id TEXT,
  production_run_id INTEGER REFERENCES production_runs(id),
  item_type TEXT,                 -- 'ingredient' | 'material'
  ingredient_id INTEGER REFERENCES ingredients(id),
  material_id INTEGER REFERENCES materials(id),
  item_label TEXT,                -- resolved ingredient/material name (display)
  qty_used REAL,
  unit TEXT,
  extras TEXT,
  notes TEXT,                     -- curated
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_prunitems_fmp ON production_run_items(fmp_id) WHERE fmp_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_prunitems_run ON production_run_items(production_run_id);
```

**Consumption posting → `inventory_txns`** (reuses the 3c-1 ledger): for each `production_run_items` row that is `item_type='ingredient'` with a non-null `ingredient_id` and a non-zero `qty_used`, INSERT-OR-IGNORE one ledger row:
- `txn_type='consumption'`, `qty = -abs(qty_used)` (negative), `unit = item.unit`,
- `txn_date = run.run_date`, `source_kind='production_run'`,
- `source_ref = 'prod_item:' || production_run_items.id` (the local row id — stable, because items are upserted idempotently by `fmp_id` so the row keeps its id; manual runs create the row once).

The `source_ref` partial-unique index (`idx_invtxn_source`, created by 3c-1) makes re-posting a no-op. **Counts use `cur.rowcount`** (real inserts), like 3c-1's seeders.

## Components & critical files

1. **`dashboard/production.py`** (new — name confirmed collision-free) — mirror `dashboard/purchase_orders.py`/`inventory.py` (reuse `_connect`/`_default_db_path`, the `_update_allowed` curated helper). Functions:
   - `init_production_schema(cx)` — the two CREATE TABLEs + indexes (idempotent).
   - `search_production_runs(q="", limit=50, offset=0, db_path=None)` — runs filtered by `batch_number LIKE` or formulation name (LEFT JOIN formulations → `formulation_name`), newest first (`ORDER BY run_date DESC, id DESC`).
   - `get_production_run(run_id, db_path=None)` — one run (+`formulation_name`) or None.
   - `list_run_items(run_id, db_path=None)` — items (LEFT JOINs resolve `ingredient_canonical`/`material_name`); include each item's posted consumption state (LEFT JOIN `inventory_txns` on `source_ref='prod_item:'||id` → `posted` 0/1).
   - `update_run_curated` / `update_run_item_curated` — `notes` only.
   - `post_consumption(cx, run_id=None, mode="all", cutoff_date=None) -> int` — post consumption for one run (or all runs when `run_id` is None). `mode` ∈ {`all`,`from_date`,`record_only`}; `record_only` posts nothing; `from_date` posts only where `run.run_date >= cutoff_date`. Returns rows inserted. Takes an open `cx`; caller commits.
   - `log_run(formulation_id, run_date, quantity_units, items, batch_number=None, db_path=None) -> int` — create a `manual` run + its `production_run_items` (each `{ingredient_id, qty_used, unit}`), then `post_consumption` for that run (`mode='all'`), commit, return run id. Raises `ValueError` on missing formulation or empty items.
   - `recipe_prefill(formulation_id, db_path=None) -> list[dict]` — convenience: returns the formulation's items (`ingredient_id`, `item_label`=ingredient name, `qty_used`=`dose`, `unit`=`dose_unit`) via `dashboard.formulations.list_items_for_formulation`, for the console to pre-fill a new run's lines (editable; no auto-scaling in v1 — operator adjusts quantities to the batch).

2. **`scripts/import_production_from_fmp.py`** (new) — reuse `from scripts.import_ingredients_from_fmp import _active,_num,_clean,_extras,_upsert` (the established helper set) and `sqlite3.Row` (Phase-3b lesson). Functions:
   - `import_production_runs(cx, rows) -> int` — FMP `production`: `formulation_id` via `id_fk_product`→`formulations.fmp_id`; `quantity_units=_num(qty)`; `run_date=production_date`; `batch_number=_clean(label)`; `status='completed'`; `source_kind='fmp'`. Upsert by `fmp_id` (`id_pk`), curated `notes` excluded from fmp_cols.
   - `import_production_items(cx, rows) -> dict` (`{"items": n}`) — FMP `production_items`: `production_run_id` via `id_fk_production`→`production_runs.fmp_id`; `item_type`/`ingredient_id`/`material_id` via `id_fk_raw`→ingredients / `id_fk_material`→materials (priority ingredient→material, per mapping 14); `item_label`=resolved name; `qty_used=_num(qty)`; `unit=_clean(unit_measurement)`. Upsert by `fmp_id`.
   - CLI `main()` with `--write` (default dry-run, rolled back) and `--consumption {all,record_only}` + `--consumption-from YYYY-MM-DD` → after import, calls `post_consumption(cx, mode=..., cutoff_date=...)`; prints run/item/consumption counts. Mirror `scripts/seed_inventory_ledger.py`.

3. **`app.py`** (modify) — add `_init_production_tables()` after `_init_inventory_tables()` (open `LOG_DB`, init, try/finally close). Add `/api/production/*` endpoints (all `@require_console_key`, `ok`/`fail`, `from dashboard import production as _prod`):
   - `GET /api/production/search` → `search_production_runs(q, limit, offset)`.
   - `GET /api/production/<int:run_id>` → `{run, items}` (404 when None).
   - `GET /api/production/recipe/<int:formulation_id>` → `recipe_prefill(...)` (for the log-run form).
   - `POST /api/production/log` → body `{formulation_id, run_date, quantity_units, batch_number?, items:[{ingredient_id,qty_used,unit}]}` → `log_run(...)`; ValueError→400; returns the new run id.
   - `PATCH /api/production/<int:run_id>` and `PATCH /api/production/items/<int:item_id>` → curated `notes`.
   - `POST /api/production/import` → multipart files `production`, `production_items` + form `write` + `consumption` (`all`|`record_only`) + optional `consumption_from`. Open `LOG_DB`, `row_factory=Row`, init ingredients+materials+formulations+production+inventory schemas, run the two importers, then `post_consumption(cx, mode=..., cutoff_date=...)`; commit on write / rollback on dry-run; return `{mode, runs, items, consumption}`. Mirror `POST /api/po/import` + the 3c-1 seed shape. **Note the importer return asymmetry** (3b lesson): `import_production_items` returns a dict `{"items": n}` — unwrap `["items"]`; the other counts are ints.

4. **`static/admin-ingredients.html`** (modify) — add `"production"` to the `labels` array + a "Production" tab mirroring the PO/Inventory tabs: search → run list (batch/formulation/date/qty); click → run detail (read-only header + line items table showing qty_used/unit and a "consumed ✓/—" badge from the `posted` flag + editable `notes`); a "Log a production run" form (formulation picker → on select, `GET /api/production/recipe/<id>` pre-fills editable line rows; qty + date; submit → `POST /api/production/log`); the Import-tab gets a "Production" section (2 file pickers + a consumption-mode selector [Post all / Record only] + optional from-date + Dry-run/Import → `POST /api/production/import`). Reuse the real `api(path,opts)` (returns `j.data`, throws) and the raw-`fetch`+`FormData` pattern for the import (NOT `api()`). **Add `display:none` initial-state CSS for the new detail panel + empty div** (recurring gotcha). Existing tabs untouched.

5. **`static/console-search-index.json`** (modify) — add `{ "title": "Production Runs", "page": "Products", "url": "/admin/ingredients", "keywords": ["production","run","batch","made","consumption","manufacture","produced"] }`.

6. **Tests** (new):
   - `tests/test_production.py` — schema; `log_run` creates a run + items AND posts negative consumption to `inventory_txns` (assert on-hand drops); `post_consumption` is **idempotent** (re-run inserts 0; a manual adjustment survives); `mode='record_only'` posts nothing; `mode='from_date'` posts only post-cutoff runs; `list_run_items` reports `posted`.
   - `tests/test_import_production.py` — importer parsing + run→formulation / item→ingredient/material resolution + idempotent re-import preserves curated `notes`; ingredient-only lines post consumption, material lines do not.
   - `tests/test_admin_production_api.py` — route-level (Pinecone `pytest.skip` pattern): log a run via the endpoint changes on-hand; search/detail/recipe endpoints return expected shapes; import dry-run returns counts.

## Reuse (don't reinvent)
- `dashboard/inventory.py` — the `inventory_txns` ledger + the INSERT-OR-IGNORE-on-`source_ref` idempotency pattern (consumption uses the SAME ledger and index).
- `scripts/import_ingredients_from_fmp.py` helpers (`_active/_num/_clean/_extras/_upsert`); `scripts/import_purchase_orders_from_fmp.py` as the importer skeleton (3-way link resolution, dict-return for items).
- `dashboard/purchase_orders.py`/`inventory.py` — `_connect`, `_update_allowed`, schema-init shape; `POST /api/po/import` + `/api/inventory/seed` as the server-side-import templates; the PO/Inventory tabs as the console template (incl. the `display:none` CSS).
- `dashboard/formulations.py` `list_items_for_formulation` for the recipe pre-fill.
- `02 Skills/fmp-loaders/mapping/13_production_runs.sql` + `14_production_run_materials.sql` as the field-mapping reference (FMP `production`: `id_pk`,`id_fk_product`,`production_date`,`qty`,`label`; `production_items`: `id_pk`,`id_fk_production`,`id_fk_raw`,`id_fk_material`,`qty`,`unit_measurement`).

## Verification (end-to-end)
1. `pytest tests/test_production.py tests/test_import_production.py tests/test_admin_production_api.py -q` — unit tests green; route tests skip locally on the Pinecone guard.
2. Against a temp `chat_log.db` (with formulations, ingredients, a baseline ledger): import synthetic production CSVs `--write --consumption all` → assert a spot ingredient's on-hand dropped by the summed `qty_used`; re-run → 0 new consumption (idempotent).
3. `--consumption record_only` on a fresh DB → runs imported, on-hand unchanged.
4. Console: log a run for a formulation (pre-filled from recipe, edited) → that ingredient's Inventory on-hand decreases; the run shows "consumed ✓".

## Build approach
Own writing-plans → subagent-driven-development cycle on branch `sess/59a2725d-p3c2` (stacked on 3c-1, since consumption writes the 3c-1 ledger), one PR. Tasks: (1) `dashboard/production.py` schema + reads + `log_run` + `post_consumption` + tests; (2) FMP importer + CLI + tests; (3) `/api/production/*` + server-side import + tests; (4) Production console tab + Import section + search index. Highest-risk = consumption posting (sign, idempotency, the mode/cutoff) and the importer link resolution — careful review. Whole-branch review at the end.

## Out of scope / deferred
- Material consumption ledger; BOM demand / reorder (3c-3); run void/reversal; recipe auto-scaling; a receipt-cutoff symmetric to the consumption cutoff (one-line follow-up if wanted).
