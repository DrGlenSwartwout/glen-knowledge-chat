# FMP Client Order History (reference import)

**Date:** 2026-06-26
**Status:** Approved (Glen: prod console + dedicated lookup view; export `clients_address`; minimal projection incl. addresses)
**Parent:** the FMP→app migration. Brings past client orders out of FileMaker into the app as read-only reference, so client order history is queryable in the console (the lookup we just did by hand).

## Problem

Past client orders live only in FileMaker (FMP). We want them as reference in the web console — "what has this client ordered before, when, for how much, to which address." The data was already exported to CSV (`/tmp/fmp-export/newapp/`); it just isn't in the app.

## Source data (already exported)

- `invoices.csv` — 425 invoices. Keys/fields used: `id_pk`, `id_fk_client`, `invoice_date`, `zcRecordStatus` (status), `zc_invoice_subtotal`, `zc_invoice_total`, `shipping_fee`, `zs_ar_os_total` (outstanding).
- `invoice_items.csv` — 3,048 line items. `id_pk`, `id_fk_invoice`, `id_fk_product`, `description`, `qty`, `price`, `zc_ext_price`.
- `clients.csv` (already loaded as `fmp_snap_clients`, 7,846) — projection uses only `id_pk`, `name_first`, `name_last`, `company`, `email`, `phone_res`, `phone_cell`, `phone_business`.
- `clients_address.csv` — 5,754 addresses. `id_pk`, `id_fk_client`, `type`, `address_street`, `address_city`, `address_province`, `address_postal_code`, `address_country`.

**Ship-to is client-level, not per-order.** FMP invoices carry no ship-to FK; `clients_address` links to the client only (no invoice FK), `type` is blank on 5,745/5,754, and 229 clients have multiple (untyped, historical) addresses. So the tool shows **all addresses on file for the client, labeled "client-level — not per-order."** True per-order ship-to (Click-N-Ship / order channel) is out of scope here.

## Privacy

The prod projection carries **only**: client names, company, email, phone, addresses, and order data (dates / amounts / line items / status). It **excludes** every clinical field in the FMP client master — `diagnose1..3`, `dob`, `gender`, `doctor`, `notes`, biofield/causal-chain data. Order amounts + names land in Render's **console-gated** DB (not public).

## Design

One projection schema, built once, used in two places (local lookup + prod push) so the lookup logic is identical.

### Component A — `dashboard/fmp_orders.py` (offline-tested)

Four projection tables (`ensure_tables(cx)`), TEXT columns:
- `fmp_clients(id_pk, name_first, name_last, company, email, phone_res, phone_cell, phone_business)`
- `fmp_invoices(id_pk, id_fk_client, invoice_date, status, subtotal, total, shipping, outstanding)`
- `fmp_invoice_items(id_pk, id_fk_invoice, id_fk_product, description, qty, price, ext_price)`
- `fmp_client_addresses(id_pk, id_fk_client, type, street, city, province, postal_code, country)`

Functions:
- `build_projection_from_csv(cx, export_dir)` — read the four CSVs, (re)populate the four tables (idempotent: `DROP`/recreate then bulk insert). Maps FMP column names → projection column names. Returns row counts.
- `client_order_history(cx, *, client_id=None, email=None, name=None)` — resolve matching client(s) (`client_id` exact; `email` case-insensitive exact; `name` LIKE over first/last/company). For each client return `{client:{id,name,company,email,phones}, addresses:[{type,street,city,province,postal_code,country}], orders:[{id,date,status,subtotal,total,shipping,outstanding,items:[{description,qty,price,ext_price,product_id}]}]}`, orders newest-first by `invoice_date`. None-raising; empty result if no match.
- `to_payload(cx)` — dump the four projection tables as JSON-able dict for the prod push.

A `__main__` / CLI builds the projection into the **local** `chat_log.db` from `/tmp/fmp-export/newapp/`.

### Component B — prod ingest (`app.py`)

`POST /api/console/fmp-orders-ingest` — `_bos_actor()`-gated. Accepts JSON `{clients:[...], invoices:[...], items:[...], addresses:[...]}` (the `to_payload` shape). `?dry_run=1` reports counts only. Real run: `ensure_tables` + replace (DROP/recreate + bulk insert) the four prod tables in `LOG_DB`. Returns counts. Run locally with curl against prod (dry first).

### Component C — console lookup (`app.py` + static)

- `GET /api/console/fmp-orders?email=|name=|client_id=` — `_bos_actor()`-gated; returns `client_order_history(...)` JSON (reads the prod projection tables). Empty/`q` too short → empty list.
- `static/console-client-orders.html` — a "Client Orders" page: search box → list of matched clients, each expandable to orders (date · status · total · outstanding, with line items) and an "Addresses on file (client-level)" block. Console look; wired into the role-based nav (`op-nav.js` / `/api/me` nav) under an owner/Rae-visible entry.

## Non-goals

- Per-order ship-to (not in FMP; belongs to the order pipeline).
- Payments table (`invoice_payments` not exported; outstanding comes from `zs_ar_os_total`).
- Linking FMP clients to the app `people` table / portals (possible later).
- Editing FMP data (read-only reference).
- Re-running the FileMaker extract from the app (CSVs are produced out-of-band).

## Error handling

- `build_projection_from_csv` / `client_order_history` none-raising per row; missing CSV → clear error in the CLI, skip in best-effort paths.
- Ingest endpoint: 401 unless `_bos_actor()`; malformed payload → 400; dry_run never writes.
- Lookup endpoint: 401 unless `_bos_actor()`; no match → empty.

## Testing

**Offline (`dashboard/fmp_orders.py`):**
1. `ensure_tables` + seed → `client_order_history` by `client_id`, by `email` (case-insensitive), by `name` (LIKE, multiple matches).
2. Orders newest-first; line items grouped under their invoice; totals/outstanding passed through; multiple addresses returned and labeled.
3. `build_projection_from_csv` on a tiny fixture CSV set → correct row counts + column mapping.
4. `to_payload` round-trips into `ensure_tables`+insert (shape stable).

**Live (post-deploy; `app.py` can't import offline):**
5. Build projection locally → `POST /api/console/fmp-orders-ingest?dry_run=1` (counts) → real → counts match.
6. `GET /api/console/fmp-orders?name=Cuddigan` → JoAnn / Sun Star Organics order(s) + address on file. Render the console page headless (assert orders + addresses render, zero console errors).

## Rollout

Local build + helper ship first (queryable on your Mac). Then deploy → run the ingest (dry → real) → the Client Orders console page is live. Re-runnable: re-export CSVs + re-ingest to refresh.
