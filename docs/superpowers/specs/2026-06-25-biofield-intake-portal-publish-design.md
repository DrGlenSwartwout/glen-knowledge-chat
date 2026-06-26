# Biofield Intake → Client Portal "Publish" Connector

**Date:** 2026-06-25
**Status:** Approved (design)
**Author:** Glen + Claude

## Problem

A completed Biofield Analysis authored in the **local** intake tool (`biofield_local_app.py`, tables `biofield_auth_*`, `biofield_narratives`) has no path to the client's **online** home. illtowell.com already has a full per-client portal — `/portal/<token>` (served by `client-portal.html`), payload from `/api/portal/<token>`, created/updated via the console-gated `POST /admin/portal/upsert`, biofield reports stored per client in `portal_biofield_reports` keyed by `(email, scan_date)`. The E4L **reveal-push** pipeline already publishes *scan*-derived content into that portal. There is no equivalent for an *authored intake* report. This connector closes that gap, reusably for every client.

## Goal

One action in the local intake tool — "Publish to portal" — that turns the authored report into the portal `content` shape and POSTs it to the prod `/admin/portal/upsert`, returning the `/portal/<token>` URL for Glen to send. PHI stays local; only the finished portal payload crosses to prod (same trust path as reveal-push).

## Non-goals

- Auto-emailing the client (Glen sends the link himself with the audio + PDF). The upsert `send` flag is left `false`.
- Self-service client login (already built, dark behind `CLIENT_LOGIN_ENABLED`).
- Populating `findings` from E4L scans (v1 leaves findings empty; intake is not scan-findings-based — enrich later).
- Any change to the prod app, the portal page, or `/admin/portal/upsert` itself. This connector only *calls* the existing endpoint.

## Design

### Field model — the portal `content` shape (target, from existing portal-seed JSON + `/api/portal` consumer)

```
content:
  greeting: str                 # "Aloha {first},"
  video: {url: "", label: "Watch your message from Dr. Glen"}
  layers: [ {n:int, title:str, meaning:str, remedy:str, dosing:str} ]
  reorder_items: [ {slug:str, qty:int, price_cents:int} ]
  pricing_note: str             # "" in v1
  findings: []                  # empty in v1
  biofield_status: "confirmed"  # authored -> un-blurred (client sees remedies)
```
The upsert call also carries top-level `email`, `name`, `scan_date`, `scan_id`, `send:false`.

### Module `dashboard/biofield_portal_publish.py` (pure, cx-based, offline-testable)

- `_load_catalog()` — reuse `dashboard.pricing._load_catalog()` (the same slug→product dict the rest of the app uses).
- `ALIAS_SLUGS` — explicit overrides applied BEFORE fuzzy resolution, for protocol wordings that differ from the catalog:
  - `"focus neuro-magnesium" -> "neuro-magnesium"` (there is no standalone "Focus" SKU; it is one product)
  - `"community spirit formula in terrain restore" -> "terrain-restore"` (compounded into the Terrain Restore base; no standalone "Community Spirit" SKU)
  Keyed by the normalized remedy string (lowercase, collapse whitespace).
- `resolve_remedy_slug(name, catalog) -> str|None` — 1) normalized exact in `ALIAS_SLUGS`; 2) else delegate to the existing in-repo resolver `dashboard.practitioner_portal.name_to_slug(name, catalog)` (exact → normalized → confidence-gated containment). Returns `None` when genuinely unresolvable.
- `_dosing(layer) -> str` — join non-empty `dosage`, `frequency`, `timing` with spaces (e.g. "1 capsule daily with food").
- `segment_narrative(narrative, layers) -> list[str]` — split the single narrative blob into one segment per layer. The narrative is written layer-by-layer; segment by locating each layer's cue (its `remedy` name, else its `head`) in order and slicing between cues. Returns a list aligned to `layers` (same length). If the cues are not found in order / cannot be aligned 1:1, return `[]` (signals fallback).
- `build_portal_content(cx, test_id, *, special_price_cents) -> dict` →
  `{ "email", "name", "scan_date", "scan_id", "content", "unresolved": [names] }`:
  1. `rep = authored_report(cx, test_id)`; `narrative = get_narrative(cx, test_id)`.
  2. `segs = segment_narrative(narrative, rep["layers"])`.
     - If `segs` is non-empty: each layer's `meaning = segs[i]`; `greeting = "Aloha {first},"`.
     - **Fallback** (`segs == []`): per-layer `meaning = ""`; `greeting = narrative` (full blob shown as the walkthrough). Never lose the prose.
  3. `layers` → `{n: layer.layer, title: layer.head, meaning, remedy: layer.remedy, dosing: _dosing(layer)}`.
  4. `reorder_items`: for each layer with a remedy, resolve slug; **dedup by slug** (so "Focus Neuro-Magnesium" yields one line); each `{slug, qty:1, price_cents: special_price_cents}`. Remedies that don't resolve are collected into `unresolved` (NOT added).
  5. `biofield_status:"confirmed"`, `video` placeholder, `pricing_note:""`, `findings:[]`.
  6. `scan_date = rep["date"]`; `scan_id = ""`.
- `publish_to_portal(payload, *, base_url, console_key, http_post=requests.post) -> dict` — POST `payload` (with `send:false`) to `{base_url}/admin/portal/upsert` with header `X-Console-Key: console_key`; return the parsed JSON (which already contains `url`/`token`). `http_post` is injectable for tests. Raises on non-2xx with the response body.

### Route in `biofield_local_app.py`

`POST /test/<test_id>/publish-portal` (console-gated, like the other local routes):
- body: `{"special_price_cents": int}` (the per-bottle courtesy price; e.g. 5000 for Karin's $50).
- builds content via `build_portal_content`; if `unresolved` is non-empty, return `{"ok":false, "unresolved":[...]}` with 409 so Glen fixes the name before publishing (no partial publish).
- else calls `publish_to_portal` with `PROD_BASE`/`CONSOLE_SECRET` from env; returns `{"ok":true, "url": ..., "unresolved": []}`.
- A "Publish to portal" button on the author/report view calls it and shows the returned URL to copy.

### Config / env (reuse existing prod-ops convention)

- `PORTAL_PUBLISH_BASE_URL` (default the prod illtowell.com / onrender base already used for prod-ops triggers).
- `CONSOLE_SECRET` (already in Doppler `remedy-match/prd`; the local app already runs under `doppler run`).

## Error handling

- Builder is none-raising on missing narrative (→ greeting fallback) and missing layers (→ empty layers).
- Unresolved remedy slugs are surfaced (route 409 + `unresolved` list), never silently dropped.
- `publish_to_portal` raises on non-2xx with the prod response body; the route catches and returns a JSON error.

## Testing (TDD, offline, tmp sqlite)

1. `resolve_remedy_slug` — alias hits (Focus Neuro-Magnesium→neuro-magnesium; Community Spirit…→terrain-restore), exact (Vitality→vitality, Nous Energy→nous-energy), and an unresolvable name → None.
2. `_dosing` — joins present fields, skips blanks.
3. `segment_narrative` — a layer-by-layer blob segments 1:1; a non-aligning blob → `[]`.
4. `build_portal_content` — seed a Karin-like test (5 layers incl. the Focus Neuro-Magnesium dedup case): asserts layer mapping, dosing concat, reorder dedup (5 layers → 5 unique slugs, one line for the combined product), `price_cents == special`, `biofield_status=="confirmed"`, meaning from segments OR fallback greeting, and `unresolved == []`.
4b. `build_portal_content` unresolved path — a remedy with an invented name lands in `unresolved` and is absent from `reorder_items`.
5. `publish_to_portal` — injected `http_post` captures the URL, header, and `send:false`; returns the prod JSON; non-2xx raises.

## Rollout

Local-only tool change + new module; no prod deploy. Go-live = run the new button for Karin (test `a3`, `special_price_cents=5000`), copy her `/portal/<token>` URL into the delivery email alongside the audio + PDF.
