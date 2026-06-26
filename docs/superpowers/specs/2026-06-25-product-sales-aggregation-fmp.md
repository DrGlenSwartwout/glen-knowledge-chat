# Product Sales Aggregation (from FMP invoices) — Design Spec

**Date:** 2026-06-25
**Status:** Design (build now). Part of the FMP→app migration arc ([[project_fmp_to_app_migration]]).
Spec re-established + grounded in the real extracted invoice schema (the earlier draft was lost in
#285's squash).

## Goal

An authoritative **`product_sales`** aggregate in the app DB (`chat_log.db`) — units + revenue per
product per month — sourced from FMP invoice line items, so "top sellers" is one query and the data
can feed reorder-demand velocity. Adds a console-gated `GET /api/console/top-products` endpoint + a
console import/view. (QBO invoices are summary-only; the SKU detail is only in FMP invoices.)

## Source (extracted + verified 2026-06-25)

`02 Skills/fmp-applescript-extract.py --source newapp --tables invoices,invoice_items` (FMP base-table
names are **lowercase** here, unlike capitalized `Production`) → `/tmp/fmp-export/newapp/`:
- **`invoice_items.csv`** — 3047 rows (3042 carry a product). The aggregation source; key columns:
  `id_fk_product` (= products' `fmp_id`), `qty`, `zc_ext_price` (line revenue, dollars),
  `zc_year`, `zc_month`, `invoice_date`, `description` (line product name), `id_pk`, `fee_name`
  (fee/non-product lines have **blank `id_fk_product`** → skip).
- **`invoices.csv`** — 424 rows (invoice-level: date, client, status). Not required for the basic
  aggregation (everything needed is denormalized onto the line item) but extracted for completeness.

Join verified: `id_fk_product` → products `fmp_id` (e.g. 425 → `microbiome`). 99/396 distinct
invoice products match products.json's 120 fmp_ids; the **prod `products` table** (181 formulations
with `fmp_id` from Phase 2) matches more; unmatched (services like "Biofield Analysis", old SKUs)
fall back to the line `description` for the name.

## Design

### Table `product_sales`

`product_fmp_id TEXT, product_slug TEXT, product_name TEXT, period TEXT ('YYYY-MM'), units REAL,
revenue_cents INTEGER, source TEXT DEFAULT 'fmp', first_sold TEXT, last_sold TEXT`, with a unique
index on **(product_fmp_id, period, source)**. Monthly grain (finest useful; the endpoint rolls up
to year/all-time; monthly enables reorder velocity).

### Ingest — `scripts/import_invoices_from_fmp.py`

Reads `invoice_items.csv`; **skips lines with blank `id_fk_product`** (fees); derives `period` from
`zc_year`+`zc_month` (fallback: parse `invoice_date`); groups by `(id_fk_product, period)`; sums
`qty`→units and `zc_ext_price`→`revenue_cents` (dollars×100, strip `$`/commas); resolves
slug/name by joining `id_fk_product` → `products.fmp_id` in `chat_log.db`, falling back to the most
common `description`. **Idempotent:** rebuild the FMP slice — `DELETE FROM product_sales WHERE
source='fmp'` then bulk insert (a re-import after a fresh extract refreshes cleanly). Dry-run vs
`--write`.

### Aggregation module — `dashboard/product_sales.py`

Pure/testable helpers: `init_product_sales_table(cx)`; `aggregate_rows(rows, resolve)` →
list of product_sales dicts (group/sum/period/cents, fee-skip); `top_products(cx, *, year=None,
by='revenue', limit=20)` → ranked list (filter `period LIKE 'YYYY%'` or all-time; rank by
revenue_cents or units). Mirrors the `dashboard/` module + console pattern of prior FMP phases.

### Endpoint + console

- **`GET /api/console/top-products?year=YYYY&by=revenue|units&limit=N`** (console-key gated) →
  `top_products(...)`.
- **`POST /api/console/sales/import`** (console-key gated, multipart upload of `invoice_items.csv`
  [+ optional `invoices.csv`], `write` flag) → runs the importer against the server's
  `chat_log.db` (dry-run counts or write) — the prod path, since prod `chat_log.db` lives on Render.
  Mirrors the existing `/api/ingredients/import` / `/api/formulations/import` console importers.
- A **Top Products** console view/tab (ranked list, year selector) + the import controls.

## Out of scope (deferred)

- Folding the app's own `orders.items_json` sales in under `source='app'` (dedup vs FMP) — a later
  increment once the app is the authoritative order source.
- Reorder-demand wiring (the `dashboard/reorder.py` consuming `product_sales` velocity) — separate.
- Service/course filtering in the endpoint (e.g. excluding "Biofield Analysis") — return everything;
  the caller can filter. Note services carry a product id but no formulation match.

## Testing (run via [[reference_deploy_chat_local_tests]])

- `aggregate_rows`: groups by (product, period); sums units + revenue_cents (cents conversion,
  `$`/comma strip); **skips blank-`id_fk_product` fee lines**; period from `zc_year`+`zc_month` and
  the `invoice_date` fallback.
- resolution: `id_fk_product` → `products.fmp_id` slug/name; description fallback when unmatched.
- `top_products`: year filter (`period LIKE`), rank by revenue vs units, limit.
- `/api/console/top-products`: console-auth (200 with key / 401 without); shape.
- import idempotency: re-import replaces the `source='fmp'` slice (no dup rows).

## Rollout

Additive (one new table); console-key gated; no public flag. **Activation:** upload
`invoice_items.csv` via the Top-Products/Import console tab → dry-run → import (the prod
`chat_log.db` gets the rows; the local `--write` only hits a local db). Re-runnable after any fresh
FMP extract.
