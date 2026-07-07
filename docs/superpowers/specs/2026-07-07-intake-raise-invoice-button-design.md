# "Raise Invoice" Button on the Biofield Intake Form — Design

**Date:** 2026-07-07
**Status:** Approved (brainstormed with Glen 2026-07-07)
**Repo:** deploy-chat (local Intake app `biofield_local_app.py` :8011 + prod console endpoints)

## Summary

On the local Biofield Intake authoring page (`/author/<id>`, Mac-only :8011), add a **"Raise invoice →"** button that creates the client's pickup invoice in one click: **Biofield Analysis as the top line**, then each authored remedy from the causal chain, priced by prod. It returns a **Print invoice** link and a summary of what was added and what was skipped. This closes the gap Glen hit with Donna Banks: an authored intake had no way to become an order/invoice without hand-entry in the console.

## Decisions (settled with Glen)

- **Approach A — one-click with sensible defaults.** The button assembles the draft and creates it; quantity/line edits happen afterward on the **Orders board** (which already supports editing). No inline quantity editor (that was the rejected approach B).
- **Biofield Analysis is always the top line** (`biofield-analysis`, qty 1). Its price comes from the client's `client_prices` courtesy (e.g. Donna's $100), applied server-side.
- **Prod is the single pricing authority.** The local app never computes prices — it POSTs lines to prod `/api/orders/manual`, which applies the courtesy + FF volume rate.
- **Non-sellable remedies are skipped, not mispriced.** A remedy name that doesn't resolve to a sellable catalog SKU (elixirs, custom blends) is surfaced in a "skipped — add manually" list rather than guessed.
- **Errors are explicit, never silent** (learning from the fee-panel silent-degrade, PR #668): an unreachable console returns a clear message.

## Reuses existing machinery

- **Order creation:** prod `POST /api/orders/manual` (`app.py:32356`) — creates a `proposed` in-house order, applies `client_prices` special prices + FF volume via `_price_inhouse_invoice`. Lines are `[{slug, qty}]`; ordering is preserved into `items_rec`.
- **Print link:** prod `GET /api/console/order/<oid>/invoice-link` (`app.py:31750`) → `/invoice/<token>?print=1` (owner-internal, no email, no Stripe).
- **Courtesy:** `client_prices` (slug `biofield-analysis`), set via the fee panel / `/api/console/client-prices`. Already the pricer's source of truth.
- **Console call pattern:** `dashboard/biofield_fee.py::_request` / `_console()` — the exact `CONSOLE_SECRET` + `PUBLIC_BASE_URL` prod-call pattern the fee panel already uses.
- **Name→product resolution:** `biofield_authoring.resolve_remedy_name` resolves a chain remedy to a catalog **product name**; the products catalog maps name↔slug.

**Not touched:** the prod order/invoice endpoints (used as-is), the fee panel, the causal-chain authoring.

## Scope

**v1 = the button + one new local route + line-assembly module + tests.** Deferred: inline quantity editor, duplicate-order hard block (the result shows the order ref so a dupe is visible/deletable in Orders), auto-emailing the invoice to the client (internal print only), editing an existing order from the intake page.

## Components

### 1. Line assembly — `dashboard/biofield_invoice.py` (new, pure + injected network)

Pure, testable core plus thin prod calls (mirrors `biofield_fee.py`).

```
BIOFIELD_SLUG = "biofield-analysis"

def resolve_line_slug(name, catalog) -> str | None
    # name -> sellable slug via the products catalog (exact/case-insensitive,
    # then the same fuzzy match resolve_remedy_name uses). None if no sellable match.

def build_invoice_lines(client, remedies, catalog) -> {"lines": [...], "skipped": [...]}
    # lines[0] is ALWAYS {"slug": BIOFIELD_SLUG, "qty": 1} (top line).
    # then one {"slug": <resolved>, "qty": 1} per remedy that resolves.
    # skipped = [remedy names with no sellable slug], preserved for display.

def default_create_order(customer, lines) -> {"ok", "order_id", "external_ref", "total_cents", "error"}
    # POST prod /api/orders/manual {customer:{name,email}, lines, pickup:true}. Explicit
    # error on non-200 / unreachable (never silent). CONSOLE_SECRET + PUBLIC_BASE_URL via _console().

def default_invoice_link(order_id) -> {"ok", "print_url", "error"}
    # GET prod /api/console/order/<id>/invoice-link -> print_url (/invoice/<tok>?print=1).
```

`_console()`, `_request()` shared with (or duplicated minimally from) `biofield_fee.py`.

### 2. Local route — `POST /author/<test_id>/invoice` (in `biofield_local_app.py`)

Mirrors `author_fee`. Injected deps `invoice_build` / `invoice_create` / `invoice_link` default to the `biofield_invoice` functions (same `create_app(...)` wiring as `fee_get/fee_set/fee_clear`).

1. Load `authored_report(cx, test_id)` → client `{name, email}` + chain remedies (names).
2. If no client email → `{"ok": False, "error": "Add a client email in the header first."}`, 400 (same guard as `author_fee`).
3. `build_invoice_lines(client, remedies, catalog)` → `{lines, skipped}`. Catalog from the local `products` import.
4. `invoice_create({"name","email"}, lines)`. On failure → `{"ok": False, "error": <explicit>}`, 502.
5. `invoice_link(order_id)` → `print_url` (best-effort; if it fails, still return success with the order ref and a note that the print link couldn't be minted).
6. Return `{"ok": True, "print_url", "external_ref", "added": [...], "skipped": [...], "total_dollars"}`.

### 3. UI — button + result in the Fee card (`render_fee_panel`, `dashboard/biofield_report_html.py`)

- A **"Raise invoice →"** button below the courtesy controls, enabled only when `state["has_email"]`.
- `raiseInvoice()` POSTs to `location.pathname + '/invoice'`, then renders into a result `<div>`:
  - a **Print invoice** anchor (`print_url`, `target=_blank`),
  - "Added: Biofield Analysis + N remedies · Total $X",
  - if `skipped`: "Not added (add manually in Orders): <names>".
- On error: the explicit error text in the result div (no silent state).

## Data flow

`/author/<id>` page → click → `POST /author/<id>/invoice` → `build_invoice_lines` (local, pure) → `default_create_order` → prod `/api/orders/manual` (prices w/ courtesy+volume, inserts proposed order) → `default_invoice_link` → prod mint link → result div shows Print link + added/skipped.

## Error handling

- No client email → inline 400, same message as the fee panel.
- Prod unreachable / non-200 on create → `{"ok": False, "error": "Couldn't reach the console to create the order."}`, 502; the UI shows it (never a silent blank).
- Create succeeded but link mint failed → success with `external_ref` shown + "print link unavailable, open it from the Orders board."
- Zero remedies resolved → still creates the Biofield-only invoice (top line always present); `skipped` lists the rest.

## Testing

Unit tests (`tests/test_biofield_invoice.py`), network injected — no prod calls:
- `build_invoice_lines`: Biofield is always `lines[0]`; a resolvable remedy becomes a line; an unresolvable one lands in `skipped`, not `lines`; ordering preserved.
- `resolve_line_slug`: exact, case-insensitive, fuzzy match; `None` for a non-catalog name.
- Route (`test_biofield_invoice_route.py`, fake injected deps): no-email → 400; happy path → `print_url` + added/skipped in the payload; create-failure → 502 with explicit error; link-mint failure → success + `external_ref` + note.

## Rollout

Local-app change; takes effect on the next `com.glen.biofield-local-server` restart (pull into `~/deploy-chat` main checkout, then `launchctl kickstart`). `render_fee_panel` edits are local-Intake-only; prod is unaffected. No feature flag (owner-only local tool).
