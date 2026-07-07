---
title: Biofield Intake fee-panel — see + set courtesy price at authoring time
date: 2026-07-07
status: design — approved with Glen, awaiting spec review
author: Claude (brainstormed with Glen)
repo: deploy-chat (local Biofield Intake app, biofield_local_app.py, :8011)
---

# Biofield Intake fee-panel

## Problem

Authoring a Biofield Analysis report and setting that client's fee happen in two
disconnected places. The report is authored in the local intake app
(`biofield_local_app.py`, `/author/<id>`, Mac-only `:8011`). The fee is a
per-client courtesy price set separately in the **prod** console at
`/api/console/client-prices`, keyed by `email` + slug `biofield-analysis`. So the
owner authors in one app, then makes a second trip to prod to set the price. The
fee-panel closes that gap: see and set the courtesy right on the authoring page.

## Decision (approved with Glen 2026-07-07)

Add a **Fee** panel to `/author/<id>` that **sees and sets** this client's
`biofield-analysis` courtesy price, writing to prod `client_prices` so the pricer
applies it when the invoice is later raised. The panel does **NOT** invoice —
console/QBO still creates the invoice (prior decision, unchanged). Scope confirmed
via the see+set option (not "see + set + raise invoice", not display-only).

### Chosen approach: local panel → prod console API

The panel reads/writes prod `client_prices` directly through the existing
`/api/console/client-prices` endpoint, reusing the cross-app pattern already in
`biofield_local_app.py` (the `/api/people` and comms helpers): `CONSOLE_SECRET` +
`PUBLIC_BASE_URL` (default `https://illtowell.com`) + an `X-Console-Key` header.
`_bos_actor()` resolves the master `CONSOLE_SECRET` first → OWNER role, so the
console key alone authenticates on that endpoint. **No prod endpoint changes.**

Rejected alternatives:
- **Local store + sync** — dual source of truth, sync drift; pricing must live
  where the pricer reads it (prod). Pure overhead.
- **Deep-link to the console pricing page** — still a second trip, just
  pre-addressed; does not close the gap.

## The panel

A **Fee** section on `/author/<id>`, placed by the header (name/email/date) since
it is client-level, not report-content.

Displays:
- **Value: $997** and **Standard charge: $300** — display only, sourced from the
  `biofield-analysis` product definition so they stay canonical (not hardcoded in
  the panel). Value falls back to the constant `997` if the product carries no
  explicit value field; standard reads the product `price_cents` (30000).
- **This client's fee** — read live from prod `client_prices` for `email` + slug
  `biofield-analysis`: either "Standard — $300" (no row) or "Courtesy — $X" with
  its note (row present).

Controls:
- **Set courtesy** — a dollar input + optional note → POST.
- **Clear → back to standard** — shown only when a courtesy exists → DELETE.
- **One-click presets** — $697 (courtesy), $100 (special), $0 (comp); these
  prefill the amount, owner still confirms with Set. Included because these are
  the recurring real amounts.
- A one-line reminder: "Applied automatically when you raise the invoice in
  console." The panel never invoices.

## Data flow

- **Read** (on page load, best-effort): `GET {base}/api/console/client-prices?email=<client>`
  with `X-Console-Key: CONSOLE_SECRET` → find the `biofield-analysis` entry in the
  returned `prices` list; None → standard.
- **Set**: `POST {base}/api/console/client-prices` body
  `{email, slug:"biofield-analysis", price_cents, note}`.
- **Clear**: `DELETE {base}/api/console/client-prices` body `{email, slug:"biofield-analysis"}`.
- Dollars↔cents conversion and request/response shaping live in a pure,
  unit-tested helper in the local app (no network in the unit tests).

## Components / boundaries

1. **Pricing client helper** — a new small module `dashboard/biofield_fee.py`
   (pure functions + one thin network fn), imported by the local app so it is
   unit-testable in isolation: `fee_get(email) -> {courtesy_cents|None, note}`,
   `fee_set(email, cents, note)`, `fee_clear(email)`, plus pure
   `dollars_to_cents`/`cents_to_dollars` and the response parser. Network calls are
   best-effort: any failure (missing `CONSOLE_SECRET`, prod unreachable, non-200)
   returns a sentinel the route renders as "pricing unavailable".
2. **Author routes** — extend the `/author/<id>` GET to fetch current fee state
   and render the panel; add `POST /author/<id>/fee` (set) and
   `POST /author/<id>/fee/clear` (delete) that call the helper and re-render/redirect.
3. **Render** — a `render_fee_panel(state)` block in the author HTML near the header.
4. **Prod `client_prices` + endpoint** — unchanged; consumed as-is.

## Error handling / edges

- **No email on the test** → panel shows "Add a client email in the header to set a
  fee," controls disabled.
- **Prod unreachable / no `CONSOLE_SECRET`** → "pricing unavailable (couldn't reach
  console)," controls disabled. Best-effort, matches existing helpers (never throws
  into the page).
- **$0 allowed** (comp); negative amounts rejected by `client_prices.set_price`
  server-side and guarded client-side.
- **Slug** is the constant `biofield-analysis` everywhere.

## Testing

- **Unit** (no network): `dollars_to_cents`/`cents_to_dollars` round-trips; the
  response parser picks the `biofield-analysis` entry out of a `prices` list and
  handles absent/failed responses; the request builder sets slug + header.
- **Render-verify** on a real authoring test that has an email: load `/author/<id>`,
  see the panel; set a courtesy, reload, see "Courtesy — $X"; clear it, see
  "Standard — $300" return. Because this writes **real prod `client_prices`**, use a
  disposable test-client email and remove the row afterward (DELETE) so no real
  client's price is altered.

## Out of scope

- Creating/raising the invoice from the intake app (stays in console/QBO).
- Per-SKU or FF-flat pricing (this panel is only the `biofield-analysis` service fee;
  the full per-client price table stays in the console).
- Any change to the pricer or the prod endpoint.

## Rollout

Local-only change to `biofield_local_app.py` (+ helper/render). The `:8011` app has
no hot-reload — restart the launchd `com.glen.biofield-local-server` (or kick the
app) to load it. No prod deploy, no flag. Reference: [[reference_biofield_intake_editor]],
[[reference_biofield_analysis_invoice]].
