# Record-Level Dashboard Deep-Links — Phase 3: Orders

**Date:** 2026-06-26
**Status:** Design approved, ready for implementation plan
**Repo:** deploy-chat (`glen-knowledge-chat`)
**Builds on:** Phase 1 (#349/#350) + Phase 2 (#352) — the citation-token registry mechanism.

## Problem

Phases 1-2 made people (→ `/console/crm?email=`) and QBO invoices (→
`/console/money?invoice=…#receivables`) clickable in the briefings. Phase 3 makes
**orders needing action** clickable in the clients-pipeline briefing, landing on
that order's card on `/console/orders` (scrolled-to + flash-highlighted), where
it can be confirmed / packed / shipped.

Orders are not in any briefing snapshot today (confirmed) — Phase 3 adds them,
the same way Phase 2 added QBO AR. Unlike Phase 2's cross-system mismatch, this
is a **clean same-table join**: both the briefing data and the `/console/orders`
board read the local `orders` table (`chat_log.db`) keyed by `orders.id`.

## Scope

**In scope:**
- Add an `orders` block (orders needing attention) to the snapshot.
- Mint **order** linkables → `/console/orders?order=<id>`.
- Surface order callouts in the **clients-pipeline** card (unpaid carts to
  recover + unshipped/backordered orders to fulfill).
- Highlight the targeted card on `/console/orders` (scroll + flash) via a new
  `?order=` autoload.

**Out of scope:**
- Payments deep-links (cut by decision — captured payments aren't actionable and
  failed charges have no retry path).
- Any change to `static/dashboard.html`, `resolveRefLinks`, or `recNavigate`
  (order links reuse them verbatim).
- The money-cash and signals-patterns cards (orders live on clients-pipeline).

## "Orders needing attention" (snapshot scope)

Only orders worth a callout enter the snapshot (only mentioned records get
links). Definition: **all open orders = every order whose status is NOT a
terminal/shipped state** (`shipped`, `delivered`, `done`), capped (~20, newest
first). This captures both flavors without fragile sub-classification:
- **unpaid** (cart / new / proposed)
- **awaiting fulfillment** (paid / confirmed / packed)
- plus any order with backordered units.

The `status` field travels into the snapshot so the LLM can distinguish "unpaid
cart to recover" from "unshipped order to fulfill." (The exact terminal-status
strings + how `list_orders` exposes status are pinned in planning by reading the
orders status model + the board's `LANES` predicates.)

## Mechanism reuse

Order links are just another registry `type` resolving to a URL. The client side
is unchanged:
- `resolveRefLinks` resolves `ref:rN` → the order URL (unknown → plain text).
- `recNavigate` already produces the correct URL for a query-only href:
  `/console/orders?order=123` → no hash, `?` present → appends `&key=` →
  `/console/orders?order=123&key=<KEY>`. The board reads `?key=` and `?order=`.

## Components

### 1. Snapshot — add `orders` (`dashboard/orders.py` + `dashboard/briefing_runner.py`)

- New pure wrapper `orders.attention_orders(limit=20)`: opens its own connection
  to `LOG_DB` (`DATA_DIR/chat_log.db`, `row_factory = sqlite3.Row`), calls the
  existing `list_orders(cx)`, filters to open orders (status not in
  `{shipped, delivered, done}`), and returns a minimal subset per order:
  `{id, name, email, status, total_cents, created_at, backorder_units}`. On any
  exception it must not crash the snapshot (the `_safe` wrapper handles that).
  - Connection helper: mirror an existing dashboard module that opens
    `chat_log.db` (e.g. `journal_store`) — exact import pinned in planning so
    `orders.py` stays consistent with the codebase.
- `gather_snapshot()` adds a **top-level** `"orders": _safe(_orders.attention_orders, label="orders")`
  (orders are not money; the clients-pipeline card owns them). `_orders` is the
  already-imported `from . import orders as _orders` (add the import if absent).

### 2. `briefing_links.py` — order linkables

- `order_url(order_id)` → `"/console/orders?order=" + quote(str(id))`. No console
  key (appended client-side).
- An order pass in `build_linkables` (after people, after invoices): for each
  `snapshot["orders"]` row with an `id`, mint
  `{type:"order", display: name or email or ("Order #" + id), url: order_url(id)}`
  and stamp `rec["ref"]`. Shared ref counter; dedup by url. `orders` may be a
  list (success) or `{"_error": …}` (skipped via `isinstance(..., list)`).

### 3. clients-pipeline prompt (`dashboard/briefing_runner.py`)

Add one instruction to `SLUG_PROMPTS["clients-pipeline"]`: call out orders to act
on from the top-level `orders` block — name them by customer + status, separating
unpaid carts (recover) from unshipped/backordered (fulfill). Add "or an order to
act on" to the RECORD LINKS example in `_build_user_prompt` so Haiku links them.
Keep orders distinct from QBO AR (which stays on the money card).

### 4. `static/console-orders.html` — card highlight

- Add `data-oid="<id>"` to the order card div in `cardHtml` (currently
  `'<div class="card">'`).
- After `load()` renders all lanes, read `?order=` from `location.search`; if
  present, find `.card[data-oid="<id>"]` (all lanes render at once, so it's in
  the DOM regardless of status), scroll it into view and add a transient flash
  class. Missing/unknown `?order=` → no-op. Add an `ord-flash` keyframe/class to
  the page styles (gold, matching Phase 2's `inv-flash`).

### 5. No change to the dashboard renderer or app.py

`static/dashboard.html`, `resolveRefLinks`, `recNavigate` untouched.
`/api/orders` and the serve path already exist.

## Data flow

```
gather_snapshot → orders = attention_orders() (open orders, minimal subset, capped)
  → build_linkables: order pass stamps ref on each order row, registry[ref] =
    {type:order, display, url:/console/orders?order=<id>}
  → LLM clients-pipeline card cites orders as [Jane, unpaid cart](ref:r7)
  → persisted to clients-pipeline.md + .links.json (Phase-1 path, unchanged)
serve → {markdown, links}
dashboard mdRender → resolveRefLinks → <a href="/console/orders?order=123" class=rec-link>
click → recNavigate appends key → /console/orders?order=123&key=<KEY>
console-orders: ?order= autoload scrolls+flashes .card[data-oid="123"]
```

## Backward compatibility

Additive. If `orders` is absent/`_error`/empty, no order linkables are minted and
the card omits order callouts — today's behavior. People + invoice links and all
existing console-orders behavior (lanes, `?q=` search) unchanged.

## Testing

- **Unit:** `order_url`; `build_linkables` order pass (id → order_url, ref
  stamped, display fallback); `_error`/empty `orders` safe; a mixed
  person+invoice+order snapshot yields all three types with shared counter + url
  dedup.
- **Real-DB test of `attention_orders`:** seed a temp `chat_log.db` (via the
  orders table schema init) with orders across statuses incl. a shipped/done one
  and a backordered one; assert the wrapper returns the open/needs-attention set
  (excludes shipped/delivered/done), the minimal field subset, and respects the
  cap.
- **Source-assert:** clients-pipeline prompt references the `orders` block;
  `console-orders.html` has `data-oid` + `?order=` autoload + flash class.
- **Render-verify (browser):** seed orders, load `/console/orders?order=<id>&key=…`,
  assert the matching card scrolls into view + gets the flash class, a
  non-existent `?order=` is a clean no-op, zero console errors.
- **Prod go-live:** after merge + deploy, trigger a regen and confirm the
  clients-pipeline registry contains `type:"order"` entries with
  `/console/orders?order=…` urls (when open orders exist).

## Open items resolved during planning

- The exact `chat_log.db` connection helper `attention_orders` should mirror
  (which dashboard module opens `LOG_DB` with `row_factory`).
- The exact terminal-status strings to exclude (read the orders status model +
  the board `LANES` predicates) and how `list_orders` rows expose `status` /
  `backorder_units`.
