# Approvals Hub (Sub-project B3) — Design Spec

**Date:** 2026-06-26
**Status:** Design approved (hub + nav group; no persistent op-nav sub-row; `/admin/membership` excluded). Ready for implementation plan.

Third increment of sub-project **B** (console consolidation). B1 (Money) and B2 (Pages) shipped. B4
(Settings stub cleanup) is a separate cycle, out of scope here.

## Goal

Give the six scattered approval queues — **Reviews, Atlas, Clips, Wholesale, Cert, Studio Credits**
— **one home and one overview**, without forcing their genuinely-different workflows into one merged
UI. A new **Approvals hub** page shows each queue as a card with a **live pending count** + an
**Open** button; op-nav groups all six under one **Approvals** entry (de-orphaning the five that have
no nav today). Each queue's own page, data, and actions are **untouched**.

**Why not a single merged inbox:** the six are heterogeneous — four different action-routing patterns
(BOS dispatch for Reviews/Studio; per-resource REST for Atlas/Clips/Wholesale; cert REST), two auth
flows, and incompatible row shapes (reviews' video/AI/gift sub-flow, cert's per-module checkbox grid,
clips' video, wholesale's license metadata, studio's approve-guard). A true aggregated inbox would
need a new fan-out backend + a generic row that can't render any of the special cases — high
complexity, low value. The hub delivers the "one place to triage" intent at low risk.
(`/admin/membership` is a subscription **grant form**, not a queue — excluded.)

## Current state (reuse-first)

| Queue | Page route | Pending-list endpoint | Pending count = | op-nav today |
|---|---|---|---|---|
| Reviews | `/console/reviews` | `GET /api/console/reviews` | `pending.length` | `data-sub="reviews"` |
| Atlas | `/admin/atlas` | `GET /admin/atlas/pending` | `(data.concepts \|\| concepts).length` | none |
| Clips | `/admin/clips` | `GET /admin/clips/pending` | `(data.clips \|\| clips).length` | none |
| Wholesale | `/admin/wholesale` | `GET /admin/wholesale/pending` | `(data.applications \|\| applications).length` | none |
| Cert | `/console/cert` | `GET /api/cert/review/list?status=submitted` | `submissions.length` | none |
| Studio Credits | `/console/studio-credits` | `GET /api/console/studio-credits` | `claims` with a pending status | none |

All six endpoints accept the console key via the `X-Console-Key` header (the `/admin/*` ones through
`@require_console_key`, which — after sub-project A — also accepts Rae's OWNER token). The `/admin/*`
**pages** additionally read `?key=` from the URL for their own data fetches (they have no gate UI).

## Design

### New page: `static/console-approvals.html` at `GET /console/approvals`

A triage hub. Console-key gated with the standard `#gate`/`unlock()`/`X-Console-Key` pattern. On load
(once unlocked), it fetches all six list endpoints **in parallel** (each with the `X-Console-Key`
header) and renders a **grid of six cards**, one per queue, each showing:
- the queue **name** + a one-line description,
- a **pending-count badge** (the count column above; on fetch error or 401, show "—" not a crash),
- an **Open** button linking to the queue's page **with `?key=<key>` appended** (required because the
  `/admin/*` queue pages read the key from the URL).

The count fetches are independent — one queue failing must not blank the others (each card resolves
on its own). A small "Refresh" control re-runs the fetches.

`data-active="bos" data-sub="approvals"` on its `op-nav.js` tag.

### Route

`@app.route("/console/approvals")` → `send_from_directory(STATIC, "console-approvals.html")`, gated +
cached like the other console pages.

### Nav grouping (`static/op-nav.js`)

- In `bosMods`, **replace** the `reviews` entry with `{ id:"approvals", label:"Approvals",
  href:"/console/approvals"+qs }`, and **remove** the five orphan entries `atlas`, `clips`,
  `wholesale`, `cert`, `studio-credits` (they're reachable from the hub now).
- In `NAV_PROFILES.glen.bos`, replace `"reviews"` with `"approvals"`. (The five orphans live in the
  owner-More group, not `glen.bos`, so they're just removed from `bosMods`.) `rae.bos` does not list
  any of these — unchanged.

### Tag the six queue pages so the BOS row highlights Approvals

- `console-reviews.html`: change its `op-nav.js` tag from `data-sub="reviews"` to `data-sub="approvals"`.
- `console-studio-credits.html`, `admin-atlas.html`, `admin-clips.html`, `admin-wholesale.html`,
  `console-cert.html`: these have **no** `op-nav.js` tag today — **add** one,
  `<script src="/static/op-nav.js" data-active="bos" data-sub="approvals"></script>`, immediately
  after `<body>` (de-orphaning them with the OPS bar). This is purely additive chrome; it does not
  change the page's own logic or its `?key=`-based data fetches.

## Out of scope

- Merging the queues' UIs, changing any queue's list/actions, or any backend/data change.
- A true aggregated inbox; a persistent op-nav sub-row of the six (the hub's card grid is the
  overview).
- `/admin/membership` (a grant form, not a queue). B4.

## Dependencies

- Sub-project A's `op-nav.js` `bosMods` + `NAV_PROFILES` (already merged) — B3 edits the `reviews`
  entry + removes the five orphan ids.
- The six existing pending-list endpoints (unchanged); all accept `X-Console-Key`.

## Testing (run via [reference_deploy_chat_local_tests])

- **Route:** `GET /console/approvals` returns 200 (console-key gated).
- **Render-verify (headless, per the render-verify lesson) — mocked endpoint data:** with the six
  list endpoints mocked to return known item counts, the hub renders **six cards** with the **correct
  per-queue counts**; each Open button's href is the right route **with `?key=` appended**; a mocked
  401/error on one queue shows "—" on that card without breaking the others; **zero JS console/page
  errors**.
- **Nav:** `node --check static/op-nav.js`; the BOS sub-row (incl. `#op-nav-more-bos`) shows
  **Approvals** and NOT `reviews`/`atlas`/`clips`/`wholesale`/`cert`/`studio-credits`.
- **Queue pages still render with the added OPS bar:** load each of the five newly-tagged pages
  (e.g. `/admin/atlas?key=…`, `/console/cert?key=…`) headless and confirm the op-nav bar renders with
  Approvals highlighted and **zero JS errors** (the page's own content/logic unchanged).

## Rollout

Additive: one new hub page + route, a small `op-nav.js` edit, and an `op-nav.js` `<script>` tag added
to five pages (+ one re-tagged). No backend/data change, no page deletions, no feature flag.
Console-key gated (and Rae's OWNER token via sub-project A).
