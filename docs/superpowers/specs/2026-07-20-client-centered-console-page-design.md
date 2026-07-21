# Client-Centered Console Page — Design

**Date:** 2026-07-20
**Status:** Approved (design), ready for implementation plan
**Repo:** `deploy-chat`

## Purpose

Assemble everything the practice already knows about a single client onto one
console page. Every piece named below already has a per-client reader somewhere
in `deploy-chat`; nobody has ever put them on one screen. This page is that
screen: a per-client hub keyed by **email** (the universal join key — there is
no single client id in the system).

The page consolidates, as **folded** (collapsible) history sections:

- **Clinical** — folds in the client's clinical tags.
- **Tests** — folded by type & date.
- **Invoices** — folded by date, each with total / paid / balance.
- **Comms** — listed by date/topic, folded.

Plus one live, non-folded **process strip**: the current test/invoice cycle
shown as sequence-status buttons, each linking to where that action happens.

## Non-goals

- No new order↔recommendation-source schema link. Orders carry no
  recommendation-source field and no FK to a recommendation record today; we do
  **not** invent one (that is a separate follow-up, "Option B", explicitly
  deferred). The process strip ties the Recommendation stage to an order
  **heuristically** — same email, most-recent active cycle — exactly as the
  existing biofield pipeline already does.
- No write/edit of client data on this page beyond following the process-strip
  links out to the existing action pages. History sections are read-only.
- No new comms/tags storage. We read existing stores; we do not create tables
  (aside from possibly extending an existing sync manifest — see Clinical Tags).

## Route & navigation

- New page: `static/console-client.html`, served by a thin route
  `@app.route("/console/client")` returning `send_from_directory(STATIC,
  "console-client.html")` with the standard no-store cache headers — matching
  the existing `bos_crm_page()` pattern (`app.py:42443`).
- The page reads the target client from `?email=` (also accept `?pq=` for
  parity with the existing `/console?pq=<email>` deep-link convention). If no
  email is present, show an email autocomplete (reuse `GET /api/people?q=`) to
  pick a client, then update the querystring.
- Nav: add a `client` sub-entry under the **People** pillar in
  `static/op-nav.js`; each page includes
  `<script src="/static/op-nav.js" data-active="people" data-sub="client">`.
- **Deep-links in:** points that today pass a client by email — the Records
  search hit (`/console/search` results, currently `"/console?pq="+email`), the
  CRM contact panel (`/console/crm`), the people directory, and the handoffs
  pipeline card — link to `/console/client?email=<url-encoded email>`. This hub
  becomes the default "open a client" destination.

## Data — existing readers reused

All readers already exist; the page assembles them. Sources:

| Section | Reader (existing) | Notes |
|---|---|---|
| Header | `GET /api/people/<id>` / `people` row (`app.py:30740`) | name, email, phone, city/state/island, profession, order_count, last_order_date |
| Clinical tags | `clinical_tags_console.client_tags(cx, client_id)` | reads `client_clinical_tags` in `e4l.db`; **no prod HTTP endpoint yet** — see below |
| Tests | `client_scans.scans_for(cx, email)` (dates+scan_id) + biofield report dates (`portal_biofield_reports.list_report_dates`) | test "type" derived from E4L finding category (`stress` vs `infoceutical`) via `biofield_e4l._group_for` |
| Invoices | live `orders` rows by email + `order_payments.balance(cx, order_id)` + historical `fmp_orders.client_order_history(cx, email)` | balance is always **derived** (`invoice − (paid − refunded)`), never stored |
| Comms | `recent_comms(cx, email)` (`dashboard/recent_comms.py`) | aggregates intake / inquiries / query_log / email feedback, each with a date + topic |
| Process strip | new resolver (below), built on `orders` columns + recommendation records | reuses `console_next_action` states and link targets |

### Bundling endpoint

Add `GET /api/console/client-360?email=` that calls the readers above and
returns one JSON payload:

```json
{
  "person": { "name": "...", "email": "...", "phone": "...", "location": "...",
              "profession": "...", "order_count": 0, "last_order_date": "..." },
  "clinical": { "name": "...", "active": [ ... ], "suggested": [ ... ] },
  "tests":    [ { "date": "2026-07-12", "type": "scan|biofield", "scan_id": ... } ],
  "invoices": { "total_paid_cents": 0, "open_balance_cents": 0,
                "orders": [ { "id": 0, "date": "...", "status": "...",
                              "total_cents": 0, "paid_cents": 0,
                              "balance_cents": 0, "edit_url": "..." } ],
                "fmp":    [ { "id": ..., "date": "...", "total": ...,
                             "outstanding": ..., "items": [ ... ] } ] },
  "comms":    [ { "date": "...", "topic": "...", "source": "intake|inquiry|query|feedback" } ],
  "process":  { "source": "biofield|scan|intake|chat|null",
                "stages": [ { "key": "recommendation", "done": true,
                              "label": "...", "action": { "kind": "link|dispatch|post",
                              "target": "..." } }, ... ] }
}
```

- **Auth:** gate exactly like the sibling per-client endpoints
  (`/api/console/client-invoice`, `/api/console/fmp-orders`) — the dominant
  inline `CONSOLE_SECRET` / `X-Console-Key` / `_owner_token_ok(key)` check.
- The page may call `/api/console/client-360` once and render everything, or the
  bundling endpoint may compose the sub-readers server-side. Prefer one bundled
  call so the page is a single fetch.

### Clinical tags — the one gap to bridge

`client_clinical_tags` lives in `e4l.db`. `e4l.db` is synced onto the prod disk
(`_db_path()` resolves `E4L_DB` env → `$DATA_DIR/e4l.db` → local), and
`clinical_tags_console.client_tags()` reads it. There is currently **no prod
HTTP endpoint** returning a client's tags (prod's `/console/clinical-tags` just
redirects to the local biofield app).

Bridge: the `client-360` endpoint (or a thin `GET
/api/console/clinical-tags?email=`) opens `e4l.db` via the same `_db_path()`
resolution and calls `client_tags(cx, client_id)`, resolving `client_id` from
email via `e4l_clients`.

**Implementation checkpoint:** verify the synced `$DATA_DIR/e4l.db` on prod
actually contains the `client_clinical_tags` table. If the sync omits it, the
tags section degrades gracefully to empty ("no tags yet") and a follow-up adds
`client_clinical_tags` to the e4l manifest sync. The page must not error when
tags are unavailable.

## Process strip — generalized resolver

The existing `_biofield_pipeline_for` (`app.py:15591`) is hardcoded to biofield
(reads `client_portals.content_json.biofield_status` + latest order, matches
`items_json LIKE '%biofield-analysis%'`). We do **not** reuse it directly. We
add a new source-agnostic resolver for the client's **current in-flight cycle**:

Stages: `Recommendation → Invoice → Sent → Paid → Fulfilled`

1. **Recommendation** — `done` if any concrete recommendation record exists for
   the email. `source` badge set from whichever exists, in priority order:
   - `biofield` — a `biofield_reveals` row (or `client_portals` biofield draft)
   - `scan` — an `ff_match_drafts` row (keyed `email + scan_date`)
   - `intake` — an `intake_responses` row with `status='submitted'`
   - `chat` — an `inquiries` / `query_log` record
   Action links to that source's action page: biofield →
   `/console/biofield-portal?email=`, scan → `/console/ff-drafts`, intake →
   the intake view, chat → the CRM/people panel. Where a source has no action
   page, the stage is informational (no button).
2. **Invoice / Sent / Paid / Fulfilled** — computed from the client's
   most-recent active (non-cancelled) `orders` row, reusing the **exact** state
   logic and link targets already in `console_next_action.py`:
   - Invoice created: a non-cancelled order exists. Two order lifecycles
     coexist — invoicing (`status in proposed/confirmed`) and fulfillment
     (`status in new/packed`); the resolver treats either as "invoice created"
     and hands the rest of the stages to the matching `console_next_action`
     resolver (`resolve_invoice` for the invoicing lifecycle, `resolve_order`
     for fulfillment). The strip surfaces whichever lifecycle the current order
     is in.
   - Sent: `invoice_sent_at` present → else action = dispatch
     `orders.send_invoice`.
   - Paid: `pay_status == 'paid'` → else action = link `/console/orders`
     ("Record payment").
   - Fulfilled: `status in (shipped, delivered, done, fulfilled)` → else action
     = link `/console/orders`.

   "Current cycle" = the latest non-cancelled order for the email (same choice
   the biofield pipeline already makes). Full invoice history lives in the
   folded **Invoices** section; the strip reflects only the active one.

The strip is rendered always-visible (not folded). Each stage shows done/pending
and, for the first not-done stage, its action button.

## Folded sections — behavior

- Each section is a `<details>`/summary (or equivalent) collapsed by default,
  with a one-line headline computed from the payload:
  - Clinical: "N active · M suggested" (or "no tags yet").
  - Tests: "N tests · latest YYYY-MM-DD" (or "no tests").
  - Invoices: "$X paid · $Y open" (from `total_paid_cents` / `open_balance_cents`).
  - Comms: "last contact <relative>" (from newest comm date).
- Expanded content lists rows newest-first. Invoices rows show total / paid /
  balance and an edit link (`/orders/new?edit_order=<id>`); FMP history is a
  sub-group below live orders. Comms rows show date + topic + source.

## Error handling & edge cases

- Unknown / no email → autocomplete picker, no error.
- Email with a `people` row but no tags / scans / orders / comms → each section
  renders its empty headline; page never errors.
- Tags unavailable on prod (e4l.db lacks the table) → Clinical section empty,
  logged, no error (see checkpoint above).
- `order_payments` balance derived per order; if an order has no payments, paid
  = 0, balance = total.
- All money in cents server-side, formatted to dollars in the page.

## Testing

- Unit: the generalized process resolver — given synthetic `orders` /
  `biofield_reveals` / `ff_match_drafts` / `intake_responses` / `inquiries`
  rows, asserts the correct `source` badge and the correct first-not-done stage
  + action target for each combination (biofield-only, scan-only, intake-only,
  chat-only, none, and multi-source priority).
- Unit: `client-360` bundling endpoint returns the expected shape for a client
  with data and for an empty client; auth returns 401 without a valid key.
- Unit: clinical-tags reader returns `{active, suggested}` from a temp `e4l.db`
  and degrades to empty when the table is absent.
- Route: `/console/client` serves the HTML with no-store headers.
- Follow the repo's existing console-test patterns (`tests/test_console_*.py`);
  respect the CI known_failures ratchet.

## Rollout

- Single PR: new resolver + bundling endpoint + tags reader + page + route +
  nav entry + deep-link updates + tests.
- No migration (Option A). No env/flag changes required beyond the existing
  `CONSOLE_SECRET` gate.

## Deferred (Option B — not in this build)

Hard order↔recommendation link: optional `recommendation_source` /
`recommendation_ref` columns on `orders`, populated going forward at every
order-creation call site, historical rows backfilled NEUTRAL/"unknown". Only
needed if the by-email/most-recent-cycle heuristic proves ambiguous in practice.
