# Reveals card → "Order remedies" button — Design

**Date:** 2026-07-02
**Driver:** Create an in-house order for a client directly from their Biofield Reveals card (immediate need: J.C. Davis needs an order to combine with Desire'e's household shipment). Generalize as a one-click owner action.

## Problem
Each client's Biofield Reveals card already carries their recommended remedies *with valid product slugs* (`d.remedies` / per-layer `remedy.slug`), but there's no way to turn those into an order from the console. The client-facing reveal page has a self-serve cart → Stripe (`/begin/biofield/<token>` "Order my remedies", visibility-gated: unpaid → top only, paid → all). The console has no owner-side equivalent.

## Goal & decisions (confirmed)
- **Where:** a button in each Reveals card's action row (`static/console-biofield-reveals.html`, `buildCard`).
- **Behavior:** one-click → creates a **proposed** in-house order via the existing `POST /api/orders/manual`. No Stripe; owner prices/sends/combines it later. Proposed orders are fully editable on the Orders board, so one click is safe (not a commitment).
- **Which remedies:** **all** remedies currently on the card (owner override — unlike the client side's unpaid→top-only gate). Read live from the card's editable slug inputs so any edits are reflected.

## Architecture — frontend only
No backend change. `/api/orders/manual` already: validates OWNER (`X-Console-Key`), resolves the client's `person_id` + saved address from email, prices via `_price_inhouse_invoice`, inserts a `proposed` order, and returns `{ok, order_id, external_ref}`. The Reveals board's `🛒 $X` order badge is computed by email match, so it lights up on the next list refresh with no extra plumbing.

### Change: `static/console-biofield-reveals.html`
1. **Button** in the action `row` (after Delete): `🛒 Order remedies`.
2. **Handler** `orderRemedies(wrap, d, btn)`:
   - Collect remedies from the card via the existing `collectRemedies(wrap)` helper → dedup slugs, drop empties → `lines = [{slug, qty:1}]`.
   - If no slugs → `alert` ("No remedies with product slugs on this reveal — add slugs first."), return.
   - Confirm. If `d.ordered` → warn ("already has an order ($X) — create another?"); else ("Create a proposed order for <name> with N remedies?").
   - Disable `btn` during the request (guard double-submit).
   - `POST /api/orders/manual` with `{customer:{email:d.email, name:d.client_name}, lines}` via the existing `api()` helper.
   - On `res.json.ok`: `loadList()` (badge updates), then offer to open it — `confirm("Order #<id> created. Open it on the Orders board?")` → open `/console/orders?order=<id>&key=<key>`.
   - On failure: `alert("Could not create order: " + error)`; re-enable button.

## Error handling / edge cases
- **No email on reveal** → `/api/orders/manual` still creates the order (email empty), but person/address won't resolve; guard: if `!d.email`, confirm "No email on this reveal — create an unlinked order anyway?".
- **No priced products** (all slugs unresolved) → endpoint returns 400 `no valid products`; surface the message.
- **Duplicate clicks** → button disabled during request.
- **Already has an order** → explicit confirm (doesn't block — a client can legitimately need a second order).

## Testing
- Frontend-only; `/api/orders/manual` is already covered by `tests/test_inhouse_order_entry.py`.
- Verify: (1) inline `<script>` passes `node --check`; (2) payload shape matches the endpoint contract (`{customer:{email,name}, lines:[{slug,qty}]}`) as exercised by the existing test; (3) live smoke after deploy = click the button on a real reveal and confirm a proposed order appears + badge lights up (this doubles as the J.C. acceptance test).

## Out of scope (YAGNI)
Quantity picker, per-remedy selection UI, shipping/address entry on the card (address auto-resolves from email; adjust on the Orders board), mirroring the client visibility gate.
