# Unify the client FF price: one number, invoice + portal

**Date:** 2026-07-08
**Context:** Rae needs to set a client's special FF (Functional Formulation) price
from the Invoice view (Comms). The invoice pricer already honors a per-client FF
flat rate (`client_prices.__all_ff__`, `ff_flat_cents`). But the composer's existing
"FF special price" field only sets the PORTAL display price (`special_price_cents`,
baked into `reorder_items` at publish) — a separate number. Result: two FF prices.
Glen chose to unify them (option b).

## Goal
`client_prices.__all_ff__` becomes the single source of truth for a client's FF
price. Rae sets it once on the Invoice panel; it drives both the invoice and the
portal reorder display.

## Changes

### 1. Portal display pricing (app.py, `api_client_portal` reorder loop)
For each `reorder_items` entry, resolve the display price with this precedence
(mirrors the invoice pricer):
1. per-item baked `price_cents` override (existing published portals keep theirs), else
2. the client's per-SKU special (`client_prices.get_price(email, slug)`), else
3. the client's FF flat (`get_ff_flat`) when the product is FF-eligible (`_qty_eligible`), else
4. the regular catalog price.
`is_special` = resolved < regular. Best-effort: a client_prices lookup failure
falls back to the current behavior (override-or-regular).

### 2. Composer: one FF field on the Invoice panel (console-biofield-portal.html)
- Remove the old "FF special price" field from the Order-remedies card and stop
  baking `special_price_cents` into `reorder_items` in `buildContent()` /
  `populate()`. New publishes carry no baked FF price; the portal reads
  `client_prices` at render.
- Add an **FF price for this client** control to the Invoice panel: loads
  `ff_flat_cents` from `GET /api/console/client-prices?email=`, saves via
  `POST {email, ff_flat_cents}`, clears via `DELETE {email, slug:"__all_ff__"}`.
  A short note: applies to remedy lines on invoices AND to the client's portal
  reorder prices; raise/edit the invoice to reprice an existing order.

## Backward compatibility (forward-only)
Already-published portals have `special_price_cents`-derived prices baked into
their stored `reorder_items.price_cents`; those keep winning at step 1, so live
portals are undisturbed. Only newly-composed/re-published portals rely on the
client_prices lookup.

## Out of scope
- The wholesale/practitioner pricing system stays separate (a practitioner's
  wholesale rate is not auto-copied into `client_prices`; Rae sets the FF flat
  explicitly, e.g. Bobbi = $50).
- No change to per-line invoice overrides or the volume curve.

## Tests
- Portal payload: FF-eligible item with no override + client FF flat set → shows
  the flat price + `is_special`; non-FF item unaffected; per-item baked override
  still wins; per-SKU client special wins over the FF flat.
- `client-prices` GET/POST/DELETE round-trip for `ff_flat_cents` (existing behavior).
