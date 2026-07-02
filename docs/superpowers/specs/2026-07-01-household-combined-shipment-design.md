# Household / Combined Shipment — Design

**Date:** 2026-07-01
**Author:** Glen + Claude (Household Orders session)
**Status:** Draft for approval
**Immediate trigger:** Combine Desire'e Dalla Guardia's and J.C. Davis's orders into one shipment to a shared address ("household"). Generalize into a reusable system for combining orders across multiple clients.

---

## 1. Problem

Two (or more) separate client orders need to ship together in one parcel to one address. Today:

- Each order lives in `bos_orders` with its own per-order `address_json`, `items_json`, and (dormant) `shipment_id` column.
- The label layer (`dashboard/easypost.py:build_shipment`) is strictly **one order → one parcel → one label**.
- Each order maps **1:1 to a QBO invoice** (per-client billing).
- A `households` feature already exists (`app.py:18405+`) that clusters *people* by shared email / phone+lastname / address+lastname, with a create/confirm API and a `households.address` field — but it is **not wired to orders or shipping**.

There is no way to group orders into one physical shipment.

## 2. Goal & non-goals

**Goal:** Let an operator combine 2+ unshipped orders into a single **combined shipment** to one shared ship-to address, buy **one label**, and have the tracking number land on **every** included order so each client still gets their own tracking email.

**Decided constraints (confirmed with Glen):**
- **Billing stays separate.** Each client keeps their own order, invoice, and payment. The combine feature lives entirely at the **shipment/fulfillment layer** — it never touches QBO invoices or commingles money.
- **Trigger = manual select + auto-suggest.** Operator multi-selects orders and clicks Combine; the board also proactively suggests combinable orders when they share a household or normalized ship-to address.

**Non-goals (YAGNI):**
- No merging of items onto a single payer / single invoice.
- No automatic combining without operator confirmation.
- No retroactive re-billing of shipping charges (see §7 — shipping consolidation is an explicit operator choice, not automatic).
- No change to the `households` people-clustering / GHL-tag logic; we only *read* it and add an order linkage.

## 3. Architecture

A new **shipment-grouping layer** sits beside orders. One `shipments` row groups N orders. Orders point back via a group id. Invoices/payments are untouched.

```
people ──person_id──► orders ──group_shipment_id──► order_shipments
  │                     │                                │
  └─ household (tags) ──┘                    ship_to_address_json (one address)
                                             tracking_number, label_url, carrier_shipment_id
```

### 3.1 Data model

**New table `order_shipments`** (created in Python at import, matching the `bos_orders` pattern in `dashboard/orders.py:_init` — no SQL migration files in this repo):

| column | type | notes |
|--------|------|-------|
| `id` | INTEGER PK | |
| `created_at` | TEXT | ISO |
| `created_by` | TEXT | operator |
| `household_id` | INTEGER NULL | FK to `households.id`, optional |
| `ship_to_json` | TEXT | the one shared ship-to address (order-address shape: `street/city/state/zip/country`, plus `name`) |
| `status` | TEXT | `open` → `packed` → `shipped` → `done` / `cancelled` |
| `tracking_number` | TEXT NULL | |
| `label_url` | TEXT NULL | |
| `carrier_shipment_id` | TEXT NULL | EasyPost shipment id |
| `notes` | TEXT NULL | |
| `updated_at` | TEXT | |

**`bos_orders` link column.** Add `group_shipment_id INTEGER NULL` (FK → `order_shipments.id`) via the existing additive `ALTER TABLE` pattern (`dashboard/orders.py:46-80`).

> Decision: add a **new** `group_shipment_id` column rather than repurpose the existing dormant `shipment_id`. The old column's intent is ambiguous (may have been meant for the EasyPost shipment id); a clearly-named new column avoids collision. The EasyPost shipment id lives on the `order_shipments` row (`carrier_shipment_id`), not per order.

An order belongs to **at most one** combined shipment. `group_shipment_id IS NULL` = ships on its own (unchanged today's behavior).

### 3.2 Module

New pure-function module **`dashboard/order_shipments.py`** (mirrors `dashboard/orders.py` style — functions over `LOG_DB`, no Flask):

- `init_order_shipments_table(conn)` — create table + additive order column.
- `create_shipment(order_ids, ship_to, household_id, created_by) -> shipment` — validates all orders exist, are unshipped, and not already grouped; stamps `group_shipment_id` on each; returns the new row.
- `get_shipment(sid)` / `list_open_shipments()` — readers; `get_shipment` hydrates its member orders.
- `add_order(sid, order_id)` / `remove_order(sid, order_id)` — adjust membership (only while `status='open'`).
- `merged_order_view(shipment) -> dict` — a **synthetic order dict** (union of all member `items`, the shared `ship_to` address, summed est. weight) shaped exactly like a real order so it can feed the existing `easypost.build_shipment` unchanged.
- `record_label(sid, tracking_number, label_url, carrier_shipment_id)` — store on the shipment **and fan out** `tracking_number` + `label_url` onto every member order (via `orders.set_order_label`), so each order's existing `send_tracking` works untouched.
- `set_status(sid, status)` — advances the shipment and its member orders together (packed/shipped/done/cancel), reusing `orders.mark_*`.
- `suggest_combinable() -> [group]` — read-only: groups unshipped, ungrouped orders that share a household slug (`_person_household_slug`) or a normalized ship-to (`street+zip+lastname`). Returns candidate clusters for the board banner.

### 3.3 Board actions (reuse `/api/action/<key>`)

Register in `dashboard/order_shipments.py` via the existing `action(key=...)` decorator (same pattern as `orders.record_payment` etc.):

- `shipments.combine` — body `{order_ids, ship_to?, household_id?}`. If `ship_to` omitted, default to household address, else the first selected order's address (operator confirms in UI). Creates the shipment.
- `shipments.add` / `shipments.remove` — membership edits.
- `shipments.create_label` — build `merged_order_view` → `easypost.build_shipment` + `buy_label` (or Click-N-Ship fallback) → `record_label` (fans out tracking).
- `shipments.send_tracking` — loops member orders, calling the existing `orders.send_tracking` per order (each client gets their own email with the shared tracking #).
- `shipments.mark_packed` / `mark_shipped` / `mark_done` / `cancel` — status transitions (cancel un-groups orders back to standalone).

### 3.4 UI (`static/console-orders.html`)

- **Row selection:** add a checkbox per order card; a floating "Combine N orders" action appears when 2+ are selected.
- **Combine modal:** shows the selected orders, their clients, and a ship-to picker (household address if found, else pick one order's address, else edit). Confirm → `shipments.combine`.
- **Combined-shipment card:** a grouped card showing member orders, the shared address, and buttons: Create label → Send tracking → Mark shipped/done, plus Remove-order / Cancel.
- **Auto-suggest banner:** on load, call a read endpoint backed by `suggest_combinable()`; if clusters exist, show "These orders could ship together as a household — Combine?" with one click to pre-fill the modal.

## 4. Data flow (happy path)

1. Operator opens `/console/orders`, checks Desire'e's order + J.C.'s order → **Combine**.
2. Modal defaults ship-to to their shared/household address; operator confirms → `shipments.combine` creates `order_shipments` row, stamps `group_shipment_id` on both.
3. Operator clicks **Create label** on the combined card → `merged_order_view` (both clients' items, one address) → `build_shipment` → `buy_label` → tracking + label stored on the shipment and **both** orders.
4. Operator clicks **Send tracking** → each client gets their own tracking email with the shared number.
5. **Mark shipped/done** advances the shipment and both orders together.
6. Money is untouched throughout — each order's invoice/payment stays exactly as it was.

## 5. Error handling & edge cases

- **Order already grouped** → `combine`/`add` rejects with a clear message ("Order #N is already in shipment #M").
- **Order already shipped/labelled** → excluded from combine; UI hides its checkbox.
- **Mixed pickup + ship** → block combine if any selected order is pickup (no label needed); surface the reason.
- **Cross-household combine** → allowed (Glen: "combine between multiple clients"). Household link is optional; if the selected orders belong to different/no households, `household_id` is null and the shared address is operator-chosen.
- **Address mismatch** → if selected orders have different ship-to addresses, the modal forces the operator to pick/confirm one; no silent default.
- **Remove last order / cancel** → deleting membership down to 0, or cancel, un-groups remaining orders back to standalone and marks the shipment cancelled.
- **Label already bought** → `create_label` is idempotent (no double-buy); re-clicking returns the stored label.
- **Partial membership after label** → adding/removing orders is blocked once the shipment leaves `open` (label bought), to avoid a parcel/label mismatch.

## 6. Testing

Pure-function module → unit-testable without Flask (matches repo's local-test pattern):

- `create_shipment` stamps `group_shipment_id` on all members; rejects already-grouped / shipped / pickup orders.
- `merged_order_view` unions items across two orders and carries the single shared address; weight sums.
- `record_label` fans tracking + label onto **every** member order.
- `set_status` advances members in lockstep; `cancel` un-groups.
- `suggest_combinable` clusters two unshipped orders sharing a household / normalized address; ignores grouped or shipped ones.
- Idempotent `create_label` (no double-buy).
- Billing untouched: no QBO/invoice/payment field on any member order changes across the whole flow (regression guard).

## 7. Shipping charges (explicit non-automation)

Combining saves postage, but each client was already invoiced their own `shipping_cents`. This design does **not** auto-adjust shipping. If Glen wants to credit the "second" client for redundant shipping, that's a manual edit via the existing `/api/orders/<id>/edit` (which already re-syncs QBO). Documented here so it's a deliberate choice, not an oversight. (Possible fast-follow: a one-click "consolidate shipping onto primary order" that zeroes the others' shipping via the edit path — deferred.)

## 8. Rollout

- Behind a flag `HOUSEHOLD_SHIPMENTS_ENABLED` (default off), consistent with recent features.
- Ship dark, render-verify on prod, then flip the flag and do the real Desire'e + J.C. combine as the go-live test.
- `render.yaml` is not the live source — no env/cron changes needed here (feature is a flag + code only).

## 9. Files touched

- **New:** `dashboard/order_shipments.py`, tests.
- **Edit:** `dashboard/orders.py` (additive `group_shipment_id` column + a couple of read helpers if needed), `app.py` (init the new table; register actions are self-registering via decorator import), `static/console-orders.html` (selection UI, combine modal, combined card, suggest banner).
- **No change:** `dashboard/easypost.py` (fed a synthetic merged order dict), `dashboard/qbo_billing.py` (billing untouched).
