# Spec: Console Biofield Portal data-entry form

**Date:** 2026-06-16
**Status:** Approved (design) — pending implementation plan
**Feature:** #3 (biofield analysis), first slice
**Scope chosen by Glen:** a console data-entry FORM to compose + publish a client's biofield portal content (Option A); include a **paste-portal-seed.json** box to pre-fill from the existing narrative skill.

---

## Goal

Replace today's hand-built JSON + curl flow with an internal console form that lets Glen/Rae compose (or load + edit) a client's biofield causal-chain content and publish it straight to their portal. It is a UI over the publish endpoint already in use — no new storage, no new customer-facing surface.

## Non-goals (YAGNI)

- FMP/E4L **auto-import** into layers (Options B/C) — the paste box is the cheap bridge instead.
- Autocomplete on the free-text layer `remedy` field (free text per the schema; reorder slugs do autocomplete).
- Live pre-publish preview (post-publish preview link only).
- Drag-reorder of layers (manual `n` ordering).
- Any feature flag — this is an internal, console-key-gated tool; publishing already does what the existing curl does.

---

## What already exists (reuse)

Confirmed by code+vault map 2026-06-16 (file:line):
- **Content model:** `client_portals` table + `dashboard/client_portal.py:25-104` — `upsert_portal(cx, email, name, content)` (mint-once token, update keeps the link), `get_portal_content_by_email(cx, email)` → `{name, content}`.
- **Content schema** (`content_json`): `{greeting, video:{url,label}, layers:[{n,title,meaning,remedy,dosing}], reorder_items:[{slug,qty,price_cents?}], pricing_note}`. Real example: `~/AI-Training/05 Clients/Brooke Webb portal-seed.json`.
- **Publish endpoint today:** `POST /admin/portal/upsert` (app.py:7463-7495) — console-key gated, `{email,name,content,send}`, optional email via `_send_full_report_email`. The new API mirrors this with form validation.
- **Render side:** `dashboard/portal_view.py:_biofield_block` reads the content; `/api/portal/<token>/view` + `static/client-portal.html` render layers/video.
- **Autocomplete sources:** `GET /api/people?q=` (console-key gated) for client lookup; `dashboard/products.py:catalog()` (slug, name, price_cents) for reorder items — exposed via the existing products API.
- **Console page pattern:** static HTML (`static/console-*.html`) + page route `/console/<x>` + API endpoint; example `/console/pricing-settings` ↔ `/api/console/pricing-settings`. Console nav in `static/op-nav.js` / console shell.

## Layer schema (fixed)

Each layer: `n` (int; 1 = surface/most-recent, increasing = deeper root), `title` (plain-English name), `meaning` (1-2 sentence clinical why), `remedy` (free-form string — product/blend description), `dosing` (protocol string). `remedy` is free text, NOT a catalog slug. Only `reorder_items[].slug` is catalog-bound.

---

## Architecture

### 1. Page route
`GET /console/biofield-portal` → `send_from_directory(STATIC, "console-biofield-portal.html")`. Console-key gated consistent with other console pages.

### 2. API — `app.py` (new endpoints, console-key gated via the existing `_console`/`_portal_console_ok` pattern)
- `GET /api/console/biofield-portal?email=<email>` → `{found: bool, name, content: {...}, has_token: bool}`. Reads `client_portal.get_portal_content_by_email`. Returns empty scaffold when the client has no portal yet.
- `POST /api/console/biofield-portal` → body `{email, name, content, send}`. Validates (below), `with _db_lock: _cp.upsert_portal(cx, email, name, content)`, optional `_send_full_report_email`. Returns `{ok, token?, url, updated, emailed}`. Mirrors `/admin/portal/upsert` semantics.

### 3. The form — `static/console-biofield-portal.html`
Vanilla JS (matches the codebase). Sections:
- **Client:** email typeahead (`/api/people?q=`) or free email + name; a **Load existing** button populates the form from `GET /api/console/biofield-portal`.
- **Paste portal-seed.json:** a textarea + "Fill form" button — `JSON.parse`, then populate greeting/video/layers/reorder_items/pricing_note. Malformed → inline error, form untouched.
- **Greeting** textarea; **Video** `url` + `label`.
- **Layers builder:** add/edit/delete rows; fields `n` (number), `title`, `meaning`, `remedy`, `dosing`. Rendered in `n` order.
- **Reorder items:** rows of `slug` (autocomplete from catalog → shows name + catalog price), `qty`, optional `price_cents` override.
- **pricing_note** textarea.
- **Actions:** *Publish* and *Publish & email client* (sets `send`). On success: show the portal URL, token (on create), and a **Preview** link to `/portal/<token>`.

### 4. Console nav
Add a "Biofield Portal" entry in the console nav/shell so it's reachable without typing the URL.

## Data flow

Open page → (Load existing by email **or** paste JSON) → edit fields → Publish → `POST /api/console/biofield-portal` → `upsert_portal` (+ optional email) → returns URL → Preview link. Idempotent token behavior identical to today (update keeps the link).

## Validation & errors

- `email` required → 400 with message.
- At least some content: ≥1 layer OR a video URL OR a greeting → else 400 "add some content".
- Each layer must have a `title` (meaning/remedy/dosing optional but encouraged); blank layers dropped on save.
- Bad/absent console key → 401.
- Invalid paste JSON → inline client-side error; never wipes the form.

## Testing

- **Route tests** (`tests/test_console_biofield_portal.py`):
  - `POST /api/console/biofield-portal` with the console key creates a portal → returns token + url; the content round-trips via `GET /api/portal/<token>` / `get_portal_content_by_email`.
  - `GET /api/console/biofield-portal?email=` returns existing content for a seeded portal, and `{found: false}` scaffold otherwise.
  - Missing email → 400; missing console key → 401.
  - `send: true` calls `_send_full_report_email` (monkeypatched) with the portal URL.
  - `GET /console/biofield-portal` serves 200.
- Form JS has no unit harness — covered by the page-served + API tests. Manual smoke: paste Brooke's seed JSON, publish to a test email, open the portal.
- Invocation + isolation per deploy-chat conventions (doppler + `deploy-chat311` venv + `DATA_DIR`; mock Supabase; tmp `DATA_DIR`). Work in the session worktree.

## Definition of done

- From `/console/biofield-portal`, a user can paste a `portal-seed.json` (or fill fields), optionally load+edit an existing client, and publish → the client's portal biofield block reflects it; the form shows the portal URL + Preview link, and "Publish & email" sends the link.
- New API is route-tested (create, load, validation, auth, send); full suite green.
