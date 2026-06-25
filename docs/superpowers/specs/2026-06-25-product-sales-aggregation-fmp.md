# Product Sales Aggregation (from FMP invoices) — Design Spec

**Date:** 2026-06-25
**Status:** Design (queued). **Execute AFTER 1b-A.** Part of the FMP→app migration arc
([[project_fmp_to_app_migration]]).
**Why now:** picking the 1b-A featured formulations surfaced that there is **no authoritative
product-level sales ranking** in the app. QBO invoices are **summary-level only** (no SKU lines);
the SKU line-item detail lives in **FileMaker (FMP) invoices**.

## Goal

Build an authoritative **aggregated product-sales table** in the app DB (`chat_log.db`) — units +
revenue per product over time — sourced from FMP invoice line items, so "top sellers this year"
(and demand planning) is a single query. Powers: featured-formulation picks, a console top-products
view, and the **PO / BOM demand calc** in the FMP→app migration.

## Source (grounded)

- **FMP** `invoices` + `invoice_items` (FK → `products`) + `invoice_payments` — **already mapped**
  in `02 Skills/fmp-loaders/manifest.toml` (`[invoices]`, `[invoice_items]`, `[invoice_payments]`,
  `[invoice_items_ingredients]`).
- **QBO is NOT usable** for this — its invoices carry totals, not SKU lines.
- Reuse the existing extract tooling: `02 Skills/fmp-odbc-extract.py` (ODBC) /
  `fmp-applescript-extract.py` (local FileMaker), and the `fmp-loaders` schema mapping.

## Dependency (blocks build)

Needs an **FMP invoice extract** (`invoices` + `invoice_items` tables via ODBC/AppleScript) — the
**same FMP re-export that gates FMP→app Phase 1**. Until that export exists, this is design-only.

## Design

- **One-time historical import:** load FMP `invoices` + `invoice_items` into staging, then
  aggregate into a **`product_sales`** table keyed by the product's stable **FMP `id_pk`** (the
  join key the migration already carries) + a period grain.
- **Table `product_sales`** (proposed): `product_fmp_id, product_slug, product_name, period (e.g.
  'YYYY' and 'YYYY-MM'), units, revenue_cents, first_sold_at, last_sold_at, source ('fmp'|'app')`.
- **Going forward:** once the app's own `orders` are authoritative, fold the app's `items_json`
  sales in under `source='app'`, **deduped by date/invoice** so the FMP history and live app sales
  never double-count. (FMP is retired after import, per the migration plan.)
- **Read surface:** `GET /api/console/top-products?year=YYYY` (console-key gated) → ranked list;
  small console view. This is the endpoint 1b-A's featured-pick question wanted.

## Out of scope

- The broader FMP→app migration (ingredients/sources, formulations, POs) — separate phases.
- Real-time sales dashboards / charts (this is the aggregate table + a ranked read).

## Testing (at build time)

- Aggregation is pure over staged rows: units/revenue per product per period; dedupe by invoice id;
  FMP↔app source separation. Golden-sample test from a small fixture of invoice_items.
- `/api/console/top-products` returns a correct ranking from a seeded `product_sales`.

## Sequence

1. **1b-A** (current) ships.
2. **FMP invoice extract** (`invoices`,`invoice_items`) — the gating prerequisite.
3. Write the bite-sized implementation plan + build this (import → `product_sales` → endpoint).
