# Phase 3c-1 Spec — Inventory Ledger + On-Hand Balance

## Context

Phase 3c is the capstone of the FMP→app-DB migration (`chat_log.db` is the authoritative source; see `00 System/claude-memory/project_fmp_to_app_migration.md`). Its goal is reorder intelligence: *what raw ingredients to buy before we run out*. That needs a live on-hand stock balance, which the app does not track today.

FileMaker already modeled this — but as **unstored calculated fields** that recompute on the fly and were never persisted or reconciled, so they drifted out of sync with reality (Glen: "I don't think it was all connected to update properly"). The FMP `ingredients` table carries: `inventory_starting` (a stored physical-count baseline), `par_level` / `par_level_unit` (reorder threshold), and the calc fields `zc_inventory` / `zc_po_qty` / `zc_production_qty` / `zc_need_purchase` (current stock, on-order, consumed, reorder flag).

**3c-1 makes that model real and live: a persisted inventory *ledger*** anchored to the baseline and updated by real events, so on-hand starts approximate and self-corrects:

```
on_hand(now) = baseline physical count (inventory_starting)
             + purchases received   (po_receiving — already imported, Phase 3b)
             − production consumed   (Phase 3c-2)
             ± manual recounts/corrections (adjustment entries)
```

3c-1 is the foundation: the ledger table, the on-hand-vs-par read model, baseline + receipt seeding from data we already hold, and a console. Production consumption (3c-2) and BOM demand / reorder (3c-3) build on it.

### Decomposition (3c, foundation-first; each its own spec→plan→SDD)
- **3c-1 (this spec):** inventory ledger + on-hand balance + baseline/receipt seeding + console.
- **3c-2:** production runs + consumption → posts `consumption` entries to this ledger (imports FMP `production` + `production_items`; needs an FMP re-export).
- **3c-3:** BOM demand (planned production × recipes) netted against on-hand + on-order → reorder list vs par/MOQ/preferred source.

## What we already have (no re-export needed for 3c-1)

- **Baseline + par** — Phase-1 ingredient import kept `inventory_starting`, `par_level`, `par_level_unit` in `ingredients.extras` (the `_extras` helper drops only `z*`-prefixed calc fields, so these stored fields survived). Read via `json_extract(extras, '$.inventory_starting')` etc.
- **Receipts** — `po_receiving` (Phase 3b): `qty_received`, `received_size` (unit string), linked to an ingredient via `po_receiving.po_item_id → po_items.ingredient_id`. The receipt event date comes from the parent PO (`purchase_orders.posted_date`, else `po_date`, else `po_receiving.created_at`).

3c-1 needs **no FMP re-export** — it seeds entirely from data already in `chat_log.db`.

## Scope

**In:**
- `inventory_txns` ledger table (one signed quantity per stock event) + a re-runnable seeding path.
- On-hand balance = `SUM(qty)` per ingredient; an on-hand-vs-par read model.
- `dashboard/inventory.py` module (reads + the one curated write: an `adjustment` entry + txn `notes`).
- Seeding: `baseline` entries from `ingredients.extras.inventory_starting`; `receipt` entries from `po_receiving` (ingredient-linked only). Idempotent by a `source_ref` key.
- Console: an "Inventory" tab in `/admin/ingredients` (levels list with below-par highlight; per-ingredient ledger; add a manual recount/adjustment) + `/api/inventory/*` endpoints, including a server-side seed endpoint (mirrors the other Phase-3 import endpoints, so it runs against the prod DB).

**Out (later phases / deferred):**
- Production consumption entries (3c-2). The `consumption` txn_type is defined in the schema now (so the ledger is complete), but nothing writes it in 3c-1.
- BOM demand, reorder list, draft POs (3c-3).
- **Material** inventory (the ledger is ingredient-scoped for v1; po_items with only `material_id` are skipped during receipt seeding).
- **Unit conversion / normalization.** Quantities are summed as-is and the unit *string* is preserved per entry; the model is deliberately approximate and self-corrects via recounts (Glen's stated approach). No g↔kg↔ea math in v1. Where a per-ingredient unit is shown, use `par_level_unit`.

## Data model (SQLite, in `chat_log.db`)

Conventions match the existing Phase-1/3b modules: `INTEGER PRIMARY KEY AUTOINCREMENT`; `TEXT DEFAULT (datetime('now'))` timestamps; `REAL` quantities; idempotency via a partial unique index used with INSERT-OR-IGNORE (SQLite cannot target a partial index with `ON CONFLICT`).

```sql
CREATE TABLE IF NOT EXISTS inventory_txns (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ingredient_id INTEGER REFERENCES ingredients(id),
  txn_type TEXT NOT NULL,          -- 'baseline' | 'receipt' | 'consumption' | 'adjustment'
  qty REAL NOT NULL,               -- SIGNED: + increases on-hand, − decreases
  unit TEXT,                       -- free-text unit string, preserved as-is (no conversion)
  txn_date TEXT,                   -- 'YYYY-MM-DD' event date (nullable)
  source_kind TEXT,                -- 'fmp_baseline' | 'po_receiving' | 'manual' | 'production_run'
  source_ref TEXT,                 -- idempotency key, e.g. 'baseline:42' or 'po_receiving:123' (NULL for manual)
  notes TEXT,                      -- curated / console-editable
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_invtxn_source ON inventory_txns(source_ref) WHERE source_ref IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_invtxn_ing ON inventory_txns(ingredient_id);
```

**Sign convention:** `baseline` and `receipt` are positive; `consumption` is negative (3c-2); `adjustment` is signed (a recount posts the *delta* needed to reach the counted value, or simply a +/− correction). On-hand is just the running sum — no special-casing per type.

**`source_ref` = the idempotency contract.** Every machine-generated entry carries a stable `source_ref`; seeding uses INSERT-OR-IGNORE on it, so re-running baseline/receipt seeding never double-posts. Manual adjustments have `source_ref = NULL` (the partial index ignores NULLs, so unlimited manual entries are allowed).

## Components & critical files

1. **`dashboard/inventory.py`** (new — name confirmed collision-free) — mirror the structure of `dashboard/purchase_orders.py` (reuse its `_connect`/`_default_db_path` pattern, the `_update_allowed` curated-write helper, `ok`/`fail` consumers upstream). Functions:
   - `init_inventory_schema(cx)` — the CREATE TABLE + indexes above (idempotent).
   - `on_hand(ingredient_id, db_path=None) -> float` — `SELECT COALESCE(SUM(qty),0) FROM inventory_txns WHERE ingredient_id=?`.
   - `inventory_levels(q="", limit=50, offset=0, db_path=None) -> list[dict]` — ingredients filtered by `name LIKE`, each with `id`, `name`, `on_hand` (correlated `SUM(qty)` subquery, default 0), `par_level` + `par_level_unit` (from `json_extract(extras,'$.par_level')` / `'$.par_level_unit'`), and `below_par` (1 when `par_level` is a number and `on_hand < par_level`, else 0). Order by `below_par DESC, name`.
   - `get_inventory(ingredient_id, db_path=None) -> dict|None` — `{ingredient: {...}, on_hand, par_level, par_level_unit, txns: [...]}` (404-able None when the ingredient doesn't exist).
   - `list_txns(ingredient_id, db_path=None) -> list[dict]` — ledger rows for one ingredient, newest first (`ORDER BY txn_date DESC, id DESC`).
   - `add_adjustment(ingredient_id, qty, unit=None, txn_date=None, notes=None, db_path=None) -> int` — insert one `adjustment` entry (`source_kind='manual'`, `source_ref=NULL`); `qty` is the signed delta; returns the new row id. Raises `ValueError` if `ingredient_id` doesn't exist or `qty` is non-numeric.
   - `update_txn_curated(txn_id, fields, db_path=None)` — curated set `{"notes"}` only, via `_update_allowed`.
   - `seed_baselines(cx) -> int` — for each ingredient with a non-null, numeric `json_extract(extras,'$.inventory_starting')`, INSERT-OR-IGNORE a `baseline` entry: `qty=inventory_starting`, `unit=json_extract(extras,'$.par_level_unit')`, `txn_date=NULL`, `source_kind='fmp_baseline'`, `source_ref='baseline:'||ingredient.id`. Returns count inserted.
   - `seed_receipts(cx) -> int` — for each `po_receiving` row whose `po_item_id` resolves to a `po_items.ingredient_id` that is non-null, INSERT-OR-IGNORE a `receipt` entry: `qty = +qty_received`, `unit = received_size`, `txn_date = COALESCE(po.posted_date, po.po_date, date(po_receiving.created_at))`, `source_kind='po_receiving'`, `source_ref='po_receiving:'||po_receiving.id`. Skip rows where `qty_received` is NULL/0 or no ingredient resolves (material-only). Returns count inserted.

   `seed_baselines`/`seed_receipts` take an open `cx` (caller commits), matching the importer convention; both must set `cx.row_factory = sqlite3.Row` defensively (Phase-3b lesson) since they read by column name.

2. **`scripts/seed_inventory_ledger.py`** (new) — CLI wrapper: open `chat_log.db`, `init_inventory_schema`, run `seed_baselines` + `seed_receipts`, print counts. `--dry-run` (default) reports how many *would* insert without writing (count candidates not already present); `--write` commits. Mirrors `scripts/import_purchase_orders_from_fmp.py`'s `main()`.

3. **`app.py`** (modify) — add `_init_inventory_tables()` (call `init_inventory_schema`) wired at module load right after the purchase-orders init; add `/api/inventory/*` endpoints (all `@require_console_key`, `ok`/`fail`, `from dashboard import inventory as _inv`):
   - `GET /api/inventory/levels` → `inventory_levels(q, limit, offset)`.
   - `GET /api/inventory/<int:ingredient_id>` → `get_inventory(...)`, 404 when None.
   - `POST /api/inventory/<int:ingredient_id>/adjust` → body `{qty, unit?, txn_date?, notes?}` → `add_adjustment(...)`; return the new on-hand.
   - `PATCH /api/inventory/txns/<int:txn_id>` → `update_txn_curated(txn_id, body)`.
   - `POST /api/inventory/seed` → server-side seed: open `LOG_DB`, init ingredients + purchase_orders + inventory schemas (the seed reads ingredients + po_receiving + po_items + purchase_orders), run `seed_baselines` + `seed_receipts`; `write` form flag → dry-run counts vs committed counts. Mirror `POST /api/po/import`'s connection/try-finally/commit shape (no file upload — it reads existing tables).

4. **`static/admin-ingredients.html`** (modify) — add `"inventory"` to the `labels` array + an "Inventory" tab button + panel, mirroring the Materials/PO tabs: debounced search → `GET /api/inventory/levels?q=` → levels list (name, on-hand, par, below-par row highlighted); click → `GET /api/inventory/<id>` → ingredient header (on-hand vs par, read-only) + a ledger table (date, type, qty, unit, notes — `notes` editable via `PATCH /api/inventory/txns/<id>`) + an "Add adjustment / recount" form (qty signed, optional unit/date/notes → `POST .../adjust`, refresh on success) + a one-time "Seed from baseline + receipts" button (calls `POST /api/inventory/seed` with a dry-run preview then write). Reuse `api()`/`escapeHtml`/`showTab`. **Add the `display:none` initial-state CSS for the new detail panel + empty div** (recurring Phase-3 gotcha — the panel must not render on load). FMP/seeded entries read-only except `notes`; existing tabs untouched.

5. **`static/console-search-index.json`** (modify) — register `{ "title": "Inventory & Stock", "page": "Products", "url": "/admin/ingredients", "keywords": ["inventory","stock","on hand","on-hand","par","reorder","balance","ledger"] }` (the URL is the ingredients console; the tab is selected within it).

6. **Tests** (new):
   - `tests/test_inventory.py` — schema + `on_hand` sums signed entries; `inventory_levels` computes `below_par`; `add_adjustment` inserts and shifts on-hand; `update_txn_curated` writes only `notes`; **`seed_baselines` + `seed_receipts` are idempotent** (run twice → second run inserts 0; on-hand unchanged) and a manual `adjustment` survives a re-seed. Use `tmp_path` + direct schema init (mirror `tests/test_purchase_orders.py`).
   - `tests/test_admin_inventory_api.py` — route-level, Pinecone `pytest.skip` pattern (mirror `tests/test_admin_po_api.py`): seed a fixture ingredient + one txn, assert `/api/inventory/levels` returns it, `/api/inventory/<id>` returns `{ingredient,on_hand,txns}`, `POST .../adjust` changes on-hand, `PATCH .../txns/<id>` 200s.

## Reuse (don't reinvent)
- `dashboard/purchase_orders.py` — `_connect`/`_default_db_path`, `_update_allowed` curated-write helper, idempotent `init_*_schema`, INSERT-OR-IGNORE-by-partial-index idempotency, the `cx.row_factory = sqlite3.Row` defensive set.
- `app.py` `/api/po/*` + `POST /api/po/import` as the endpoint + server-side-seed templates; the Materials/PO tabs in `static/admin-ingredients.html` as the UI template (including the `display:none` CSS rule).
- `dashboard/__init__.py` `require_console_key`, `ok()`, `fail()`.

## Verification (end-to-end)
1. `pytest tests/test_inventory.py tests/test_admin_inventory_api.py -q` — unit tests green; route tests skip locally on the Pinecone guard (run in CI).
2. Against a temp `chat_log.db` seeded with a few ingredients (with `extras.inventory_starting` + `par_level`) and a couple of `po_receiving` rows: run `scripts/seed_inventory_ledger.py --dry-run` → review candidate counts; `--write` → assert `on_hand` = baseline + received for a spot ingredient; an ingredient below par shows `below_par=1`.
3. Re-run `--write` → 0 new rows, on-hand unchanged (idempotent).
4. Add a manual adjustment via the console (e.g. recount −0.5 kg) → on-hand drops; re-run the seed → the adjustment persists.
5. Load `/admin/ingredients?key=...`, open the Inventory tab, confirm levels render with below-par highlight, open an ingredient, see its ledger, add an adjustment.

## Build approach
Own writing-plans → subagent-driven-development cycle on branch `sess/59a2725d-p3c1`, off `main`, one PR. Tasks: (1) `dashboard/inventory.py` schema + reads + curated write + tests; (2) `seed_baselines`/`seed_receipts` + `scripts/seed_inventory_ledger.py` + idempotency tests; (3) `/api/inventory/*` endpoints + server-side seed + route tests; (4) Inventory console tab + search-index. Whole-branch review at the end. The seeding task (idempotency, the receipt join, the baseline `json_extract`) is the highest-risk and gets careful review.

## Out of scope / deferred
- Production consumption entries (3c-2) — the `consumption` type exists in the schema; nothing writes it yet.
- BOM demand / reorder list / draft POs (3c-3).
- Material inventory; unit conversion/normalization; multi-location stock.
