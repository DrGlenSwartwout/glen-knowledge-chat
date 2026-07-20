# Invoice Line-Item Notes + Saved-Message Library + Total Units to Ship

**Date:** 2026-07-20
**Repo:** deploy-chat
**Status:** Design approved, pending spec review

## Summary

Three related additions to the in-house invoice / order flow:

1. **Per-line customer-visible note.** Each line item on an invoice gets a free-text
   note that the customer sees on the invoice.
2. **Shared saved-message library.** Notes are picked from, and auto-saved to, one
   app-wide library. A manage control lets the owner delete stored messages.
3. **Total units to ship.** The New Order form shows the summed quantity of all line
   items at the top of the Line Items section, above the Qty column.

All three are owner-only, reusing the existing RBAC on the invoice edit endpoint. No
changes to pricing, QBO, or the existing order-level `invoice_note`.

## Background / current state

- In-house invoices are rows in the unified `orders` table (`dashboard/orders.py`).
  Line items are **not** a separate table — they are a JSON array in
  `orders.items_json`. Each line is a dict: `{slug, name, qty, unit_cents, line_cents,
  ...optional override/service/kind/tier/gift/desc}`.
- The line-item record is built in `_price_inhouse_invoice(...)` (`app.py:39777-39881`),
  specifically the `rec = {...}` at `app.py:39870`.
- The invoice editor + New Order form is a single file: `static/order-new.html`
  (create mode by default; edit mode when URL has `?edit_order=<oid>`, JS const
  `EDIT_OID`). Table header at `order-new.html:94`; row renderer `renderLines()` at
  `order-new.html:309-321`; POST builder `linesPayload()` at line 326; edit-load
  mapping at lines 478-479.
- Edit endpoint: `POST /api/orders/<oid>/edit` (`app.py:40022-40069`), owner-only,
  routed through `_reprice_and_persist_invoice(...)` (`app.py:39960-40013`) →
  `_bos_orders.upsert_order(...)`.
- There is already an **order-level** customer-facing note, `invoice_note` (one per
  invoice), distinct from the internal-only `notes` column. Our new field is the
  **per-line** analog and is genuinely new.
- Reusable-value store pattern to mirror: `dashboard/client_prices.py` — a small keyed
  SQLite table with self-healing `init_table(cx)` and add / list / delete CRUD. No SQL
  migration file (that mechanism is Postgres-only, `migrations/*.sql`, not relevant here).

## Requirements

### R1 — Per-line note field (data)
- Add a `note` string key to each line record in `_price_inhouse_invoice`
  (`app.py:39870`). Default empty string / omitted when blank. It persists
  automatically inside `items_json` — **no DB migration** for the note itself.
- Round-trips through save (create + edit) and load-for-edit.

### R2 — Per-line note field (New Order form / editor UI)
- Extend the client `LINES` model in `order-new.html` with a `note` field.
- In `renderLines()`, each row gains: a **text input** bound to that line's note, and
  an adjacent **dropdown** listing the shared library's saved messages. Selecting a
  saved message fills the text input for that line.
- `linesPayload()` includes `note` for each line.
- Edit-load (lines 478-479) reads `note: l.note || ""` from `o.items`.
- The dropdown options are loaded once on page load from `GET /api/invoice-snippets`.

### R3 — Shared saved-message library (store + auto-save + delete)
- New SQLite table `invoice_line_snippets`:
  `id INTEGER PK, text TEXT NOT NULL UNIQUE, created_at TEXT, last_used_at TEXT`.
- New CRUD module `dashboard/invoice_snippets.py` modeled on `client_prices.py`:
  `init_table(cx)`, `add(cx, text)` (INSERT ... ON CONFLICT: touch `last_used_at`),
  `list_all(cx)`, `remove(cx, snippet_id)`. `init_table` invoked before first use, same
  pattern as the other `init_*_table` calls.
- **Auto-save on invoice save:** in both the create and the edit server-side save
  paths, after the line list is built, iterate line notes and `add(cx, note)` each
  non-empty note. New notes get inserted; existing ones just touch `last_used_at`.
  One shared library used by every line.
- Endpoints (owner-only, same RBAC as the edit endpoint):
  - `GET /api/invoice-snippets` → `{snippets: [{id, text}]}` for dropdown population.
  - `DELETE /api/invoice-snippets/<int:sid>` → removes one stored message.

### R4 — Manage / delete UI
- One "Manage saved messages" control near the Line Items header (shared library ⇒ one
  control, not one per row). It lists stored messages, each with a ✕ that calls the
  DELETE endpoint and refreshes the in-page dropdowns.

### R5 — Total units to ship
- On the New Order form, display **"Total units to ship: N"** at the top of the Line
  Items section, above the Qty column — the sum of all line `qty` values.
- Recomputed whenever a qty changes, hooked into the existing `renderLines()` /
  recompute path so it stays in sync on add / remove / qty edit. Present in both create
  and edit mode (same file).

### R6 — Customer-facing display
- The customer sees line items on exactly **one** surface: the tokenized invoice page
  `GET /invoice/<token>` → `static/invoice.html`. Rendering is **client-side** (server
  ships JSON; the page builds the DOM). There is **no** server-side PDF — "Save as PDF"
  is a browser print of the same client HTML. The client portal and the
  "invoice ready" email only link out to `/invoice/<token>`; they render no line items,
  so they need **no change**.
- The server applies a **key whitelist** when building the customer payload:
  `_invoice_line_view(l)` (`app.py:41003-41040`) copies only allowed keys. The `note`
  will NOT reach the customer unless we add `out["note"] = l.get("note")` there. It is
  then included via `_invoice_summary` (`app.py:41092-41119`, `lines` at 41104) and
  served by `GET /api/invoice/<token>` (`app.py:41134-41169`).
- Client render, two places in `static/invoice.html`:
  - **On-screen rows** — `renderLines()` (`invoice.html:337-351`, row at 347-348).
    But `note` must first survive the local re-map of `ORDER.lines` → `LINES`, which
    uses the same whitelist and happens in **several** handlers: `load()`
    (`invoice.html:198-200`), plus the `pushUpdate` / `applyPlan` /
    `toggleInvoiceMembership` handlers (`430-431`, `479-480`, `498-499`). Add `note`
    to each `.map(l => ({...}))` or it is dropped before `renderLines` sees it.
  - **Printable rows** — `buildInvoicePrintHtml(o)` (`invoice.html:286-324`, rows at
    290-298). This reads `o.lines` (i.e. `ORDER.lines`) directly, so it only needs the
    server whitelist fix (step above); no `LINES` re-map involved.
- Model the note markup/placement on the existing order-level `invoice_note`:
  on-screen "Notes" card at `invoice.html:144-147` + populate at `:205`; print block
  `pd-note` CSS at `:86-88` + emit at `:321`. Show the note on/under its line row; omit
  cleanly (no empty label) when a line has no note.
- **Out of scope but noted for later consistency** (not customer-facing, also loop
  line items): Rae's composer `/api/console/client-invoice`
  (`app.py:39231-39280`, lines at 39260-39264) and `static/practitioner-client.html`.
  Not changed by this work.

## Non-goals / YAGNI
- No per-product or per-invoice scoping of the library (explicitly one shared list).
- No edit-in-place of stored messages (delete + re-add covers it).
- No changes to pricing, discounts, QBO push, shipping, or the order-level
  `invoice_note`.
- No search/pagination on the library (a simple list; prune via delete).

## Data flow

```
New Order form / editor (order-new.html)
  page load ── GET /api/invoice-snippets ──> populate per-row dropdowns + manage list
  owner types note OR picks from dropdown ──> LINES[i].note
  qty change ──> renderLines() ──> "Total units to ship" recompute
  submit ──> linesPayload() includes note per line
       │
       ▼
create path / POST /api/orders/<oid>/edit  (app.py, owner-only)
  build line list (rec.note at app.py:39870)
  for each non-empty note: invoice_snippets.add(cx, note)   ← auto-save
  upsert_order(items_json includes note)
       │
       ▼
customer invoice surface(s)  ── render each line's note  (R6 targets)
```

## Testing
- Round-trip: create an invoice with line notes → reopen in edit mode → notes present.
- Auto-save: a new note appears in `GET /api/invoice-snippets` after save; a duplicate
  note does not create a second row.
- Delete: `DELETE /api/invoice-snippets/<id>` removes it; dropdowns refresh.
- Total units: sum matches across add / remove / qty edit.
- Customer surface: the note shows on the customer invoice view for a line that has one,
  and is absent (no empty row/label) for a line that doesn't.
- Owner-only: non-owner gets the same RBAC rejection as the existing edit endpoint.

## Rollout
- Single worktree branch off deploy-chat default; PR per repo policy. Deploying
  deploy-chat causes a brief single-instance outage, so batch/merge per the usual
  deploy-window discipline.
