# Console Cross-links & Affordances (Sub-project C3) — Design Spec

**Date:** 2026-06-26
**Status:** Design approved. Final slice of sub-project **C** (console inline actions). Follows C1
(reorder → draft PO) and C2 (money actions).

## Goal

Four small, independent, pure-frontend affordances that let an operator act/navigate where the item
is shown — reusing existing endpoints, no new backend:
1. **Reveals → Portal link** — open the Portal editor pre-loaded with a reveal's client.
2. **CRM queue Cancel / Retry** — per-item actions on the GHL sync queue.
3. **CRM contact search** — a name/email type-ahead to find a contact without knowing the exact email.
4. **Dashboard → right section + key-fix** — a "Money & Cash" alert lands on the Money board's
   Receivables tab, and the console key stops getting dropped on the way.

**Deferred (not C3):** true record-level dashboard deep-links (jump to invoice #1234). The Intelligence
briefings are LLM-generated markdown with **no machine-readable record ids**, so that needs a 3-layer
change (briefing prompt emits structured refs → frontend parser → per-record anchors on each target
page) — its own medium-large project.

## Current state (reuse-first)

- **Reveals:** `static/console-biofield-reveals.html` `buildCard()` renders each reveal; the row data
  (`/api/console/biofield-reveals`) reliably carries `d.email`. The Portal page
  `static/console-biofield-portal.html` `loadExisting()` calls `GET /api/console/biofield-portal?email=`
  — the API already accepts `?email=` — but `boot()` (~line 141) does NOT seed `#email` from the URL.
- **CRM:** `static/console-crm.html` `loadQueue()` (~line 113) renders `/api/ghl/queue/pending` items
  read-only (`{id, op, email, status, ...}`). `POST /api/ghl/queue/result` `{id, status, result}`
  exists and is gated by the same `X-Console-Key` the page sends via `hdr()` (~line 84) — it does a bare
  `UPDATE status=?` with no restriction, so Cancel (`status:"cancelled"`) and Retry (`status:"pending"`)
  work with **zero backend change**. The page has a bare `#email` input (~line 59), no search. The
  Portal's `clientSearch()` (console-biofield-portal.html ~line 157) is a working type-ahead over
  `GET /api/people?q=<term>&limit=8` → a dropdown → fills `#email`/`#name` — clone it.
- **Dashboard:** `static/dashboard.html` `ACT_AREA` (~line 974) maps intelligence-callout slugs to
  board URLs; `actNavigate()` (~line 987) does `location.assign(href + sep + "key=" + key)`. Two
  problems: `money-cash → "/console/finance"` 302-redirects to `/console/money#receivables` and the
  appended `?key=` is **dropped on the redirect** (you land on the gate); and `signals-patterns →
  "/console/home"` also redirects. Also, `href + "?key="` mis-places the key when `href` contains a
  `#` fragment (`/console/money#receivables?key=…` puts the key in the fragment).

## Design

### Component 1 — Reveals → Portal link (2 files, pure frontend)

- `static/console-biofield-portal.html` `boot()`: after `loadCatalog()`, read the URL email and
  auto-load — `var e = new URLSearchParams(location.search).get('email'); if(e){ $('email').value = e;
  loadExisting(); }`.
- `static/console-biofield-reveals.html` `buildCard()`: append an **"Open Portal →"** anchor to the
  card's button row — `href = '/console/biofield-portal?email=' + encodeURIComponent(d.email) + '&key='
  + encodeURIComponent(key())`, `target="_blank"`. (Carries the console key so the portal loads
  authenticated.)

### Component 2 — CRM queue Cancel / Retry (1 file, pure frontend)

In `console-crm.html` `loadQueue()`, render two buttons per queue item:
- **Cancel** → `POST /api/ghl/queue/result` with `{id: q.id, status: "cancelled"}` (via `hdr()`).
- **Retry** → same endpoint with `{id: q.id, status: "pending"}` (so the drain re-picks it up).
After either, reload the queue (`loadQueue()`). Use the page's existing `hdr()`/`key()`. No backend
change. (Pass `q.id` as a number in the onclick — no JSON-in-onclick of objects.)

### Component 3 — CRM contact search (1 file, pure frontend)

Add a name/email type-ahead to `console-crm.html`'s `#email` input, cloning the Portal's
`clientSearch()` pattern: on input, `GET /api/people?q=<term>&limit=8` → render a small dropdown of
matches (name + email) → clicking one fills `#email` (and any name field) and hides the dropdown.
Reuse the page's `key()`/auth. Debounce the input (the Portal's pattern already does); hide the
dropdown on outside-click.

### Component 4 — Dashboard section deep-link + key-fix (1 file, pure frontend)

In `static/dashboard.html`:
- **`ACT_AREA`:** `money-cash → "/console/money#receivables"` (the Money board's Receivables tab — its
  own hash-tab logic activates it); `signals-patterns → "/console"`; keep `clients-pipeline → "/console"`,
  `shaira-daily → "/console"`. (No more lossy `/console/finance` / `/console/home` redirects.)
- **`actNavigate()`:** insert the key **before** any `#fragment`. Split the href on `#`; append
  `?key=<key>` (or `&key=`) to the path part; re-append the fragment. Result: `/console/money?key=…#receivables`
  — key preserved in the query, the Money board lands on Receivables. This fixes the key-loss bug for
  every callout, not just money-cash.

## Out of scope

- Record-level dashboard deep-links (deferred — see Goal).
- A new GHL queue retry endpoint (the existing `/api/ghl/queue/result` suffices; no thin wrapper).
- Any change to the Portal/CRM/Dashboard data, the reveal/portal/queue/people endpoints, or biofield
  logic.

## Dependencies

- The existing `/api/console/biofield-portal?email=`, `/api/console/biofield-reveals`,
  `/api/ghl/queue/pending`, `/api/ghl/queue/result`, `/api/people?q=` endpoints (all unchanged), and
  the Money board's `#receivables` tab (from B1).

## Testing (run via [reference_deploy_chat_local_tests])

All pure-frontend → **headless render-verify (the render-verify lesson), mocked endpoints, zero JS
console/page errors:**
- **Reveals→Portal:** with `/api/console/biofield-reveals` mocked to a reveal carrying `email`, the
  card shows an **Open Portal →** anchor whose href is `/console/biofield-portal?email=<enc>&key=<enc>`.
  Separately, loading `/console/biofield-portal?email=jo@x.com&key=…` (with the portal-detail endpoint
  mocked) seeds `#email` and calls `loadExisting()` (the detail fetch fires for that email).
- **CRM Cancel/Retry:** with `/api/ghl/queue/pending` mocked to one item, the row shows Cancel + Retry;
  clicking Cancel posts `/api/ghl/queue/result {id, status:"cancelled"}` and Retry posts
  `{id, status:"pending"}`, then reloads.
- **CRM search:** typing in `#email` fires `/api/people?q=…`; a mocked match renders a dropdown; clicking
  it fills `#email`.
- **Dashboard:** a `money-cash` callout's `actNavigate` produces `/console/money?key=<enc>#receivables`
  (key before the fragment); a `signals-patterns` callout → `/console?key=…` (no redirect).

## Rollout

Purely additive frontend across four files (`console-biofield-portal.html`, `console-biofield-reveals.html`,
`console-crm.html`, `dashboard.html`). No backend, no schema, no new endpoint, no feature flag.
Console-key / OWNER-token gated (existing).
