# Phase 3 — In-Funnel Sales Pages: AI Product Image Generation

**Date:** 2026-06-18
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render service `glen-knowledge-chat`)
**Builds on:** Phase 1 (live) + Phase 2 (live, PR #172). Parent: `2026-06-18-funnel-sales-pages-design.md`.

---

## Problem

The product page's images section (#7) is an empty placeholder. Phase 3 generates real product imagery per product — grounded in the formula's hero ingredients and mechanism — and displays it, so every page has compelling visuals. The pairwise picker / credit / champion-challenger tournament is Phase 4; Phase 3 generates + stores the images and shows one per type.

## Scope

**Generate:** **2 variants per type** = 4 images per product — Mode A (botanical-lifestyle) ×2 and Mode B (mechanism) ×2. **Display:** one botanical + one mechanism image as product imagery in the images section. Generating both variants now makes Phase 4's pairwise pick a pure UI/logic add (images already exist).

**Gate:** new flag **`SALES_PAGES_AI_IMAGES`** (default OFF). Ships dark on top of live Phase 1/2.

**Architecture:** Render-side, end-to-end. Generation runs in the existing background scheduler (off web workers); Flux 1.1 Pro via Replicate's REST API; images saved to Render's persistent disk and served via a route. No Mac dependency.

**Out of scope (later phases):** pairwise pick UI, the 👍/👎 or pick credit, the champion-challenger tournament + challenger regeneration (all Phase 4).

---

## Components (small, isolated)

### `dashboard/sales_images.py` — data layer + queue
SQLite (`chat_log.db`):
- `sales_page_images(id, product_slug, kind, variant, filename, state, created_at)` — `kind ∈ {botanical, mechanism}`, `variant` 1..2, `state ∈ {ready, failed}`. (Phase-4 adds `wins/losses/role` columns later.)
- `sales_image_queue(product_slug PK, state, requested_at, updated_at)` — `state ∈ {pending, done, failed}`, mirrors `dashboard/process_queue.py`.
- Functions: `init_tables(cx)`, `enqueue(cx, slug)`, `list_pending(cx)`, `mark_done/mark_failed(cx, slug)`, `record_image(cx, slug, kind, variant, filename)`, `get_images(cx, slug) -> list`, `display_images(cx, slug) -> {botanical: filename|None, mechanism: filename|None}` (the first ready variant per kind).

### `dashboard/sales_image_prompts.py` — pure prompt builder
- `build_image_prompts(product) -> {"botanical": [p1, p2], "mechanism": [p1, p2]}`. Mode A grounds in the formula's fresh + powder botanical ingredients in a natural kitchen scene (mature woman, herb garden); Mode B renders the mechanism (e.g. a cell with a radiant protective field repelling the formula's key stressor), derived from the product name + ingredient list + a short mechanism phrase. Two variants per type differ by an explicit variation directive (angle / styling) so they're genuinely distinct for the Phase-4 A/B. Pure, no I/O — unit-testable.

### `dashboard/replicate_client.py` — Replicate REST client
- `generate_image(prompt, *, token=None) -> bytes`: POST `https://api.replicate.com/v1/predictions` for Flux 1.1 Pro (`black-forest-labs/flux-1.1-pro`), poll `get` until `succeeded`, download the output image bytes. Uses `requests` (already a dep) — NO new package. Reads `REPLICATE_API_TOKEN` from env. Raises on failure/timeout (caller handles).

### Background worker (in the existing scheduler)
A job registered in `_run_cron` (app.py ~L15093): each tick, `list_pending()` → for each queued slug, build the 4 prompts, call `replicate_client.generate_image` ×4, save bytes to `DATA_DIR/sales-images/<slug>/<kind>-<variant>.png`, `record_image(...)`, then `mark_done`. On any image failure, `mark_failed` + log; partial successes are still recorded. Bounded: one slug per tick (or a small N) to cap Replicate spend per cycle.

### Serving route
`GET /begin/product-image/<slug>/<filename>` → `send_from_directory(DATA_DIR/sales-images/<slug>, filename)` (mirrors the `/clips` route at app.py:11849). 404 if absent.

### Page-data + frontend
- `begin_product_page_data`: when `SALES_PAGES_AI_IMAGES` is on, the `images` section body becomes `{"images": [{kind, url}], "state": "ready"|"generating"|"none"}` — `display_images()` populates URLs (`/begin/product-image/<slug>/<file>`) for ready images; if the queue row is pending → `state:"generating"`; if neither → `state:"none"`. Flag off → unchanged (empty placeholder, no new fields).
- `static/begin-product.html` `renderImagesBody`: render the ready images (one botanical + one mechanism). If `state:"none"`, on first open POST `/begin/product-image-gen/<slug>` (enqueue) and show a muted "Generating product imagery…" placeholder, then poll `/begin/product-page-data` every few seconds until images appear (bounded retries). If `generating`, show the placeholder + poll. NO emoji.
- New route `POST /begin/product-image-gen/<slug>` → `enqueue(cx, slug)` (flag-gated; 404 if off / unknown slug). Idempotent.

---

## Data flow

1. Viewer opens the images section → page-data `state:"none"` → frontend POSTs enqueue + shows "Generating…".
2. Scheduler tick picks up the slug → builds 4 prompts → Flux ×4 → saves to disk → records rows → `mark_done`.
3. Frontend poll re-fetches page-data → `state:"ready"` → renders one botanical + one mechanism image. Cached forever; only first-open per product triggers generation (~$0.16 Flux, one-time).

## Error handling

- Replicate failure/timeout → that image skipped + logged; queue `mark_failed` if all four fail (re-enqueueable). The images section shows nothing on total failure (page never breaks — imagery is enhancement).
- Missing `REPLICATE_API_TOKEN` → worker logs + marks failed, no crash. (Go-live needs the token in Render env.)
- Disk write failure → logged, image skipped.
- Concurrent enqueues idempotent on `product_slug`.

## Testing

- **Prompt builder:** both modes present; grounded in product name + ingredients; the two variants per type differ. Pure, no I/O.
- **Data layer:** enqueue/list_pending/mark_done idempotent; `record_image` + `get_images`/`display_images` return first ready per kind.
- **Worker:** with a mocked `replicate_client.generate_image` (returns fake bytes) and a tmp `DATA_DIR`, processing a queued slug writes 4 files, records 4 rows, marks done; a raising mock marks failed and doesn't crash.
- **Replicate client:** with `requests` mocked (predictions create → succeeded → image bytes), returns bytes; error path raises.
- **Page-data marker:** flag on + ready → `images` urls + `state:ready`; pending → `state:generating`; none → `state:none`; flag off → no images field (Phase-1/2 identical).
- **Serving + enqueue routes:** 200/404; enqueue idempotent; flag off → 404.
- Follow deploy-chat test isolation (tmp `$DATA_DIR/chat_log.db`; mock Supabase; `importorskip` playwright). Visual render is a manual pass.

## Flags

- `SALES_PAGES_ENABLED` (live) + `SALES_PAGES_AI_COPY` (live) + **`SALES_PAGES_AI_IMAGES`** (new, default OFF). Go-live also needs **`REPLICATE_API_TOKEN`** in Render env. With the images flag off, pages behave exactly as today.
