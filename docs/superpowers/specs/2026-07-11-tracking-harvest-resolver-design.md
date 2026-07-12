# CNS Tracking Watcher — Harvest Resolver (close the blank-To gap for good)

**Date:** 2026-07-11
**Status:** Design approved (pending spec review)

## Problem

The CNS tracking watcher (`cns_tracking_watcher.py`) drafts a USPS tracking email
per shipment, resolving the recipient email by matching the CNS "Shipped To" name
to a GHL contact (`dashboard/ghl.py::find_contact_by_name`). When no contact
matches, it leaves the draft's `To:` blank and marks it `needs_review` — a
deliberate safety choice (never mail the wrong person on a fuzzy match).

The recurring blank-To cases are buyers who simply aren't in GHL yet:
- **eProcessing / Authorize.net phone-email orders** (Rae charges a saved card) —
  these NEVER hit the GrooveKart `#801` onboarding webhook, so they will keep
  producing blank-To drafts forever.
- **Storefront first-timers** whose `#801` webhook failed to fire/onboard.

Today each requires a manual lookup + fill (done ~10× by hand this session).

## Goal

When the ship-to name doesn't resolve in GHL, **harvest** the buyer's real
contact from the order emails already in the connected mailbox, add them to GHL
(so future orders resolve automatically), and — only when precision-safe — fill
the draft `To:` so the first order needs no manual touch either.

Strict no-regression: any case that isn't precision-safe behaves exactly as today
(blank To, `needs_review`).

## Approach

Add a harvest step to `handle_confirmation` at the exact point it currently gives
up (no GHL match). Keep the core decision function pure by **injecting** a
`harvest_fn` callable, mirroring the existing injected `find_contact` / `draft_fn`
seams — so all logic stays unit-testable with fakes, no Gmail/GHL network.

### New module: `dashboard/order_harvest.py`

- `parse_order_email(source, body) -> {name, email, phone, address, products}`
  Pure parser per source. **Always reads the CUSTOMER block, never the merchant
  block** (Healing Oasis / 351 Wailuku Drive / support@remedymatch.com /
  (808) 217-9647). Sources:
  - `eprocessing` — "Approved Transaction" (Transactions@prod.eprocessingnetwork.com):
    customer `Name:` + `E-Mail:` block.
  - `authorizenet` — "Merchant Email Receipt" (noreply@mail.authorize.net):
    billing/shipping + email.
  - `neworder` — remedymatch "New order : #NNNN" (support@remedymatch.com):
    `customer: NAME (EMAIL)` + shipping/billing + product line items. **Storefront.**
  - `invoice` — in-house Remedy Match invoice: `To:` header = buyer email.

- `harvest_buyer(gmail_search, ship_to_name) -> hit | None`
  Searches the mailbox for order emails near `ship_to_name`, parses candidates,
  and returns a hit **only if precision-gated**:
  1. `normalize(customer_name) == normalize(ship_to_name)` in a structured source, AND
  2. exactly **one** distinct customer email across all matching candidates.
  Zero matches, a name mismatch, or ≥2 distinct emails → `None`.
  The hit carries `{email, first, last, phone, source, products}` (source is the
  winning source key, used for the onboarding decision below).

### The precision gate is the whole safety story

Ship-to ≠ buyer cases (Jeff Slayter ships to "Kate Gray"; Mary Gaughan
drop-ships to son "Matthew") produce no customer-block named after the ship-to,
so `harvest_buyer` returns `None` → blank To, `needs_review`, exactly as today.

### On a precise harvest (inside `handle_confirmation`)

1. `contact_id, created, err = ghl_upsert_contact(email, first=ship_to_first,
   last=ship_to_last, phone, source_tag=<by source>)`
   - first/last set to the **ship-to name** so `find_contact_by_name` resolves
     future orders. Idempotent (GHL upserts by email).
   - `source_tag`: `source:gk-purchase` for `neworder`; `source:phone-email-order`
     for `eprocessing`/`authorizenet`/`invoice`.
2. **Onboarding decision:** if `source == "neworder"` AND `created is True`
   (a genuine first-time storefront buyer the webhook missed) → run the full
   `ghl_add_to_pipeline` + `ghl_enroll_workflow` (i.e. `ghl_onboard_contact`'s
   pipeline+workflow steps). Otherwise **records-only** — never re-onboard an
   existing contact or a phone-email client (would re-blast the new-customer
   sequence at established clients like Morris / Pam Schreur).
3. Fill `To:` = harvested email; `status = "drafted"`; `match_confidence =
   "harvested"`.

### On no precise harvest

Unchanged: `To:` blank, `status = "needs_review"`.

### Coupling note

GHL-add and To-fill happen together under the one gate. The
"add-to-GHL-but-To-blank" case only arises without an email, and without an email
there's nothing useful to add (email is both the GHL upsert key and the value the
resolver returns). So: precise harvest **with an email** → do both; otherwise →
`needs_review`.

## Wiring

- `cns_tracking_watcher.py` gains a thin `make_gmail_search_fn(service)` that runs
  `service.users().messages().list/get` for a query and returns parsed bodies —
  injected into `handle_confirmation` as `harvest_fn` (a closure binding the
  gmail search + `ghl_upsert_contact` + onboarding). Core stays pure.
- `handle_confirmation(html, msg_id, cx, find_contact, draft_fn, harvest_fn=None,
  dry_run=True)` — new optional `harvest_fn`; when `None`, behavior is identical to
  today (safe default; keeps existing call sites/tests green).

## Dry-run

Harvest may **search read-only** to preview a resolution, but performs **no**
`ghl_upsert`/onboarding, no drafts, no DB writes. Dry-run output shows the
would-be resolution + whether it would onboard.

## Testing (all fakes, no network; synthetic/redacted fixtures — no real customer PII committed)

- `parse_order_email` per source: correct customer name/email extracted; merchant
  block ignored; missing-email → email `None`.
- `harvest_buyer` gate: exact single-email → hit; name-mismatch → None;
  two-distinct-emails → None; merchant-block-only → None; ship-to≠buyer → None.
- `handle_confirmation`:
  - no GHL match + precise harvest (eprocessing) → To filled, `ghl_upsert` called,
    records-only (no onboarding), status `drafted`.
  - no GHL match + precise harvest (neworder, created) → onboarding invoked.
  - no GHL match + precise harvest (neworder, existing) → records-only.
  - no GHL match + no harvest → blank, `needs_review` (regression guard).
  - GHL match → unchanged (harvest never called).
  - `harvest_fn=None` → identical to pre-change behavior.

## Out of scope

- The ~395 lead/list broken-name GHL contacts (separate hygiene sweep).
- Merging duplicate contacts (done separately this session).
- Changing the CNS parse or the draft template.
