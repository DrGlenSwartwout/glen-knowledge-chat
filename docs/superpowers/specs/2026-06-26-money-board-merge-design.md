# Money Board Merge (Sub-project B1) — Design Spec

**Date:** 2026-06-26
**Status:** Design approved (pure merge, named "Money"). Ready for implementation plan.

First increment of sub-project **B** (console consolidation / de-sprawl) from the console IA
review. B1 merges the two money boards into one. The other B increments — **B2** (merge the 4 AI
page-editors), **B3** (merge the approval queues), **B4** (Settings stub cleanup) — are separate
spec→plan→build cycles, out of scope here.

## Goal

Collapse the two separate money boards — **Payments** (Stripe ledger) and **Finance** (QBO
receivables) — into a single **Money** board with two tabs, so the daily "who paid / who owes"
question lives in one place instead of two nav entries. **Pure consolidation:** every existing
action and view is preserved exactly; no new actions (the money inline-actions belong to
sub-project C). Backends are untouched.

## Current state (reuse-first)

- **`static/console-payments.html`** at `GET /console/payments` (app.py ~24678): read-only Stripe
  transaction ledger. Calls **`/api/payments`**. Source filter buttons (All / Subscriptions /
  Coaching / Trials / Funnel), summary stats, and a failed-charge banner. An `unlock()` console-key
  gate.
- **`static/console-finance.html`** at `GET /console/finance` (app.py ~24843): QBO accounts-
  receivable aging. Calls **`/api/finance/ar`** for the data and **`/api/action/finance`** (the BOS
  action dispatch) for **Refund / Send reminder / Void**. An `unlock()` gate.
- Both pages load `op-nav.js` with `data-active="bos" data-sub="payments"` / `"finance"`.
- The dashboard's "Money & Cash" intelligence card deep-links to **`/console/finance`** (so that URL
  must keep working).

## Design

### New page: `static/console-money.html` at `GET /console/money`

One page, two tabs, reusing the existing backends verbatim:

- **Tab "Payments"** — the Stripe ledger lifted from `console-payments.html`: the `/api/payments`
  table, the All/Subscriptions/Coaching/Trials/Funnel source filters, the summary stats, and the
  failed-charge banner. Read-only.
- **Tab "Receivables"** — the QBO AR lifted from `console-finance.html`: the `/api/finance/ar`
  aging table and its **Refund / Send reminder / Void** actions (via `/api/action/finance`).

Behavior:
- The page loads `op-nav.js` with `data-active="bos" data-sub="money"`.
- A single `unlock()` gate at the page level (the existing pattern), shared by both tabs.
- Each tab **lazy-loads** its data on first activation (don't fetch Receivables until its tab is
  opened). The active tab is reflected in the URL hash (`#payments` / `#receivables`) so it is
  deep-linkable and survives reload; default tab = Payments.
- The two pages' scripts are lifted into the two tab panels. Any colliding top-level name
  (`unlock`, `load`, filter state, etc.) is namespaced per tab so the two scripts coexist on one
  page without clobbering each other.

### Backend

**Unchanged.** `/api/payments`, `/api/finance/ar`, and `/api/action/finance` keep their exact
signatures and auth. The merge is front-end only. A new route serves the page:
`@app.route("/console/money")` → `send_from_directory(STATIC, "console-money.html")`, gated like the
other console pages (mirror `/console/payments`'s route exactly, incl. its `Cache-Control`).

### Nav (`static/op-nav.js`)

In `bosMods`, replace the two entries `{id:"payments"…}` and `{id:"finance"…}` with one
`{id:"money", label:"Money", href:"/console/money" + qs}`. In the `NAV_PROFILES` map (added in
sub-project A), replace `payments`/`finance` with `money` in both `glen.bos` and `rae.bos` (Money is
a Rae-primary board). No other op-nav change.

### Old routes redirect (preserve bookmarks + the dashboard deep-link)

- `GET /console/payments` → **302 redirect** to `/console/money#payments`.
- `GET /console/finance` → **302 redirect** to `/console/money#receivables`.

Replace the two existing `send_from_directory` route bodies with the redirects. (The dashboard's
"Money & Cash" card points at `/console/finance`; after this it lands on the Receivables tab.) The
old `console-payments.html` and `console-finance.html` files are **deleted** once their markup/JS is
lifted into `console-money.html` — nothing else serves them, and their content now lives in the
tabs.

## Out of scope

- Any new action (failed-charge retry/contact, record-payment on a receivable) — sub-project **C**.
- Changing the Stripe or QBO data, the `/api/payments` / `/api/finance/ar` / `/api/action/finance`
  endpoints, or the refund/void/reminder behavior.
- The Hawaii GET tax surfaces (handled under Settings → Tax in sub-project A).
- B2/B3/B4.

## Dependencies

- Sub-project A's `op-nav.js` `NAV_PROFILES` map (already merged) — B1 edits the `payments`/`finance`
  entries in it.
- The existing `/api/payments`, `/api/finance/ar`, `/api/action/finance` endpoints.

## Testing (run via [reference_deploy_chat_local_tests])

- **Route:** `GET /console/money` returns 200 + the page (console-key gated like its siblings);
  `GET /console/payments` and `GET /console/finance` return 302 to `/console/money#payments` /
  `#receivables`.
- **Render-verify (headless, per the render-verify lesson) — the core gate:** load
  `/console/money?key=<console-key>` and assert, with **zero JS console/page errors**:
  1. Both tabs render; the Payments tab is active by default and shows the Stripe ledger + the
     source-filter buttons + (if present) the failed-charge banner.
  2. Switching to Receivables lazy-loads the AR table and shows the **Refund / Send reminder / Void**
     controls.
  3. `/console/money#receivables` deep-links straight to the Receivables tab.
  4. The two lifted scripts coexist — no `ReferenceError`/redeclaration, filters and actions wired.
- Confirm `op-nav.js` still passes `node --check`; the BOS sub-row shows one **Money** entry (not
  Payments + Finance), and a Rae-profile render keeps Money primary.

## Rollout

Additive + redirect: a new page + route, two route bodies swapped to redirects, the two old HTML
files deleted, and a small `op-nav.js` edit. No backend or data change, no feature flag. Console-key
gated as today (and Rae's OWNER token, via sub-project A's auth-gate work).
