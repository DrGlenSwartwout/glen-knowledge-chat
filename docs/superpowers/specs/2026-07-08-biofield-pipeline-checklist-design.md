# Biofield pipeline — the per-client sequential checklist

**Date:** 2026-07-08
**Context:** The manual $300 Biofield service spans Glen's local intake app and Rae's
prod console. Glen and Rae need one page that shows every client's progress through
the process and the next click, from request to fulfillment.

## The page
Grow the existing Handoffs page (Communication → **Biofield Pipeline**) into a
per-client checklist. One card per in-flight client; each card lists the six
sequential steps with a status dot (done = green, next = highlighted, todo = amber)
and a link or button to do that step.

### Steps (in order)
1. **Paid — $300 Biofield Analysis** — a paid, non-cancelled biofield-analysis order → link: Orders
2. **Intake authored & handed off** — a biofield portal exists (the first prod-visible intake signal; authoring itself is in Glen's local app) → note "awaiting Dr. Glen's intake app" when absent
3. **Analysis published & emailed** — portal `biofield_status == confirmed` → button: composer Publish & email
4. **Invoice published** — latest order `portal_published` → button: composer Invoice panel
5. **Invoice paid** — order `pay_status == paid` → link: Orders
6. **Fulfilled — remedies shipped** — order status shipped/delivered/done → link: Orders

"Complete" = analysis published AND invoice paid AND fulfilled (a pre-paid,
analysis-only client never publishes a remedy invoice, so step 4 is not a completion
criterion). Completed clients hidden by default; a "Show completed" toggle passes `?all=1`.

## API
`GET /api/console/biofield-pipeline` (console-gated, `?all=1`) → `{ok, clients:[{email,
name, updated_at, done_count, complete, steps:{<key>:{done, ...}}}]}`. Population =
union of clients with a biofield portal and clients with a non-cancelled
biofield-analysis order (catches paid-but-not-handed-off, e.g. Steve Fox). Status is
computed from prod (`client_portals` + `orders`) — see `_biofield_pipeline_for`.

## The cross-app seam
Prod cannot see Glen's local intake authoring; the pipeline tracks a client from paid
/ handed off onward, and shows "awaiting Dr. Glen's intake app" for the authoring step
until the handoff creates the portal draft. Everything Rae acts on is fully live.

## Tests
- Handed-off client: paid + handed_off done, analysis_published false (ai_draft), in flight.
- Completed client hidden by default, shown with `?all=1`.
- Paid-but-not-handed-off (Steve shape): appears, paid done, handed_off false.
- Endpoint requires the console key.
