# Product Page Images — Phase A: Variation- & Model-Tagged Gallery

**Date:** 2026-06-20
**Status:** Draft for review
**Feature flag:** `SALES_PAGES_IMAGE_VARIATIONS` (new, ships dark)

## Context

This is **Phase A** of a 3-phase effort to turn the in-funnel product/sales-page
images into a self-improving, split-tested system.

- **Phase A (this spec):** show **4 images per type** (botanical + mechanism = 8/product),
  each generated from a distinct **prompt variation** and a rotating **image model**,
  with every image tagged by both so later phases can attribute performance. Plus a
  grid layout, a model-source label, and a lazy backfill action.
- **Phase B (later spec):** "pick your favorite" on the gallery → writes to the existing
  `sales_page_votes`, attributing each vote to the chosen image's prompt variation **and** model.
- **Phase C (later spec):** aggregate votes **per variation and per model across all products**
  → leaderboard + champion-challenger evolution (retire losers, promote challengers) + console management.

Phase A is the only user-visible piece on its own; its two tags (`prompt_variant_id`,
`model_id`) are the hinge that B and C hang on with **no rework**.

## Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| "Type" of image | The 2 style modes: **botanical**, **mechanism** |
| Images per type | **4** (8 per product) |
| Variation source | A **prompt-variation registry** — 4 active variations/kind, meaningfully different scenes (not just lighting) |
| Models | **Multi-engine**, starter pool: **Flux 1.1 Pro + Google Imagen 4 + Recraft V3** (all ~$0.04/img via Replicate) |
| Assignment | Each product covers **all 4 variations per kind**; **models rotate** per slot with a per-product offset → balanced (variation × model) coverage across products → clean marginal stats for both axes |
| Layout | **Grid per type**: 4-across desktop / 2×2 mobile, labeled sections |
| Model label | Subtle per-image caption ("made with Flux 1.1 Pro"), **on by default**, hideable via a simple env/setting (console toggle UI deferred to Phase C) |
| Win signal (Phase B) | **User picks/votes** (reuse `sales_page_votes`) |
| Backfill | **Lazy** — new products get 8 going forward; existing products topped up via an **admin-triggered** action |
| Flag | New `SALES_PAGES_IMAGE_VARIATIONS` gates new generation **and** new grid display; OFF = current behavior |

## Current state (from code exploration)

- Route `/begin/product/<slug>` → `static/begin-product.html`; data `/begin/product-page-data/<slug>` (`app.py` ~3165–3327). Phase-3 image branch ~3258–3276.
- `dashboard/sales_images.py`: table `sales_page_images(id, product_slug, kind, variant, filename, state, created_at)`. `display_images()` returns the **first ready image per kind** (1/kind). `list_ready()` returns all ready (ordered kind, variant). `record_image()`, `enqueue()`, `next_variant()`.
- `dashboard/sales_image_prompts.py`: `IMAGE_KINDS=("botanical","mechanism")`; `_STYLES` already has **4 styles/kind**; `build_image_prompts()` currently emits **2/kind** (`range(2)`); `build_one_prompt(kind, variant_index)` cycles styles.
- Generation: Replicate (Flux 1.1 Pro) via a background worker over `list_pending()`; gen endpoint `app.py` ~3790–3801. Existing **fallbacks** Flux → Imagen/OpenAI exist to build the engine abstraction on.
- Template render: `renderImagesBody()`/`renderImages()` (`begin-product.html` ~700–768) assumes 1 image/kind.
- Flags: `SALES_PAGES_AI_IMAGES` (live, Phase 3), `SALES_PAGES_IMAGE_PICK` / `_TOURNAMENT` (dark). **Pick/tournament are out of scope** and untouched.

## Architecture (Phase A)

### 1. Prompt-variation registry — `dashboard/sales_prompt_variations.py` (new)

```
sales_prompt_variations(
  id INTEGER PRIMARY KEY,
  kind TEXT,                       -- 'botanical' | 'mechanism'
  label TEXT,                      -- short human label, e.g. 'kitchen-flatlay'
  prompt_template TEXT,            -- full scene body; NO product names, NO _NO_TEXT (appended at gen)
  state TEXT DEFAULT 'active',     -- 'active' | 'retired'
  created_at TEXT, retired_at TEXT
)
```
- Auto-create + **seed 4 active variations per kind** from code defaults if the table is empty (same pattern as other tables here).
- Seed variations differ in **scene/composition**, not just lighting — each is a complete, visibly distinct scene. **Exact seed prompt strings are finalized during implementation, bound by this principle (4 visibly distinct scenes per kind).** Example botanical variations: (a) wooden-counter flat-lay of whole botanicals, (b) mature woman preparing herbs with garden behind, (c) close intimate herb still-life, (d) overhead market-basket abundance. Example mechanism variations vary the cell/field metaphor (single glowing cell with shield; particle inflow; cross-section with luminous membrane; field repelling a stressor).
- `active_variations(cx, kind) -> [rows]` ordered by id.

### 2. Model/engine registry + dispatcher — `dashboard/sales_image_models.py` (new)

```
sales_image_models(
  id TEXT PRIMARY KEY,             -- 'flux-1.1-pro' | 'imagen-4' | 'recraft-v3'
  label TEXT,                      -- 'Flux 1.1 Pro' (display caption)
  engine TEXT,                     -- 'replicate'
  engine_ref TEXT,                 -- replicate model identifier
  state TEXT DEFAULT 'active',     -- 'active' | 'retired'
  created_at TEXT
)
```
- Seed the 3 starter models.
- **Dispatcher** `generate(model_id, prompt, *, aspect) -> bytes|url`: maps `model_id` → `engine_ref`, calls Replicate, normalizes output. Reuses/extends the existing Flux/Imagen/OpenAI fallback code. On engine error, falls back to the baseline model and records which model actually produced the image.
- `active_models(cx) -> [rows]` ordered by id.

### 3. `sales_page_images` — add tags

Add nullable columns (legacy rows = NULL = untagged/legacy):
```
ALTER TABLE sales_page_images ADD COLUMN prompt_variant_id INTEGER;
ALTER TABLE sales_page_images ADD COLUMN model_id TEXT;
```
`variant` stays as the per-product slot index (1..4 within a kind). `record_image()` extended to persist both tags.

### 4. Generation: balanced (variation × model) assignment — `sales_image_prompts.py` / `sales_images.py`

New `build_generation_jobs(cx, slug) -> [job]` producing **4 jobs per kind** (8 total), each:
```
{ kind, variant(slot 1..4), prompt_variant_id, model_id, prompt_text }
```
Assignment rule (deterministic, balanced, testable):
- For kind K, slot i (0..3): `variation = active_variations(K)[i]` → **all 4 variations covered every product**.
- `model = active_models()[(i + offset(slug)) % len(models)]`, where `offset(slug)` is a stable per-product integer (e.g. `crc32(slug) % len(models)`) → model–variation pairings **rotate across products** so each axis gets balanced marginal coverage.
- `prompt_text = variation.prompt_template + " " + _NO_TEXT` — the variation template already encodes the full distinct scene/look (no separate lighting suffix), and still carries **no product names / no text** (the existing anti-text rule).

`enqueue()` writes these 8 jobs to the pending queue carrying their tags + prompt; the worker calls the dispatcher per job and `record_image()` stores the result with `prompt_variant_id` + `model_id`. Skips slots already present (idempotent — drives lazy backfill).

### 5. Display — endpoint + `display_images_grouped()`

`dashboard/sales_images.py`: new `display_images_grouped(cx, slug, per_kind=4) -> {kind: [ {url, model_id, model_label, prompt_variant_id} ]}`:
- Returns up to 4 **tagged** ready images per kind, ordered by `variant`.
- **Graceful fallback:** if a kind has no tagged images yet (legacy product not backfilled), return its legacy untagged image(s) with `model_label=None` (cosmetic only; excluded from Phase B/C attribution).

`app.py` Phase-3 branch (~3258–3276), **only when `SALES_PAGES_IMAGE_VARIATIONS` is on**: replace `display_images()` with `display_images_grouped()` and emit the grouped shape into the images section payload. Flag off → unchanged current behavior.

### 6. Template — grid per type + model label

`begin-product.html` `renderImagesBody()`/`renderImages()`: when the payload is grouped, render two labeled sections ("Botanical", "Mechanism"), each a responsive grid (CSS: `grid-template-columns` 4-across ≥768px, 2×2 below). Each tile = image + subtle model-source caption when `model_label` present. Graceful with <4 tiles. Add a small scoped CSS block.

### 7. Backfill — admin action

`POST /admin/sales-images/backfill` (auth-gated, console-driven), arg = one slug or `all`:
- For each target product, call `build_generation_jobs` and enqueue only the **missing** (kind, slot) cells up to 8 → tops legacy products up to the full tagged set. Idempotent; safe to re-run. You trigger it, controlling when Flux/Imagen/Recraft cost lands (~$0.32/product).

### 8. Feature flag

`SALES_PAGES_IMAGE_VARIATIONS` (env, read at startup like the others). OFF = today's behavior (2/kind, single-per-kind display). ON = registries active, generation produces the tagged 8, grid shows up to 4/kind with labels. The backfill action works regardless (lets you pre-warm images before flipping the flag).

## Data flow

```
admin backfill / new product
  → build_generation_jobs(slug): 8 jobs, each tagged (variation, model)
  → enqueue → worker → dispatcher.generate(model_id, prompt)
  → record_image(... prompt_variant_id, model_id) into sales_page_images
visitor → /begin/product/<slug>
  → /begin/product-page-data → display_images_grouped → grouped {botanical:[...], mechanism:[...]}
  → template renders 2 labeled 4-tile grids + model captions
```

## Out of scope (designed-for, not built here)

- **Phase B:** the pick UI + writing votes; `sales_page_votes` already exists. Attribution = join vote.chosen_variant → the image's `prompt_variant_id`/`model_id`.
- **Phase C:** cross-product aggregation, prompt/model champion-challenger replacement, console leaderboard/management. No schema rework needed — the registries carry `state` (active/retired) and images carry both tags.
- Pick/tournament (`SALES_PAGES_IMAGE_PICK`/`_TOURNAMENT`) code — untouched.

## Testing (TDD, pytest, follows existing sales-test patterns; honor `feedback_deploy_chat_test_isolation`)

1. Registry seeds: 4 active variations/kind; 3 active models; idempotent seed.
2. Prompt build: `prompt_text` includes `_NO_TEXT`, contains **no product name/number**.
3. Assignment: `build_generation_jobs` returns 8 jobs (4/kind); every kind covers all 4 variations; models rotate by per-product offset; deterministic for a given slug; balanced across a set of slugs.
4. `record_image` persists `prompt_variant_id` + `model_id`.
5. `display_images_grouped`: ≤4 tagged/kind ordered by variant; legacy fallback when untagged; includes `model_label`.
6. Endpoint returns grouped shape only when flag on; unchanged when off.
7. Backfill enqueues only missing slots; idempotent; respects 8 cap.
8. Dispatcher maps `model_id`→ref; falls back + records actual model on engine error.
   Mock Replicate/Supabase; seed a tmp DB; no live network.

## Risks / to confirm before/at build

- **Live pricing & availability** of Imagen 4 and Recraft V3 on Replicate (and exact `engine_ref`s) — confirm against the account's Replicate token before wiring; pricing in this spec is early-2026 ballpark.
- Aspect-ratio parity across engines (1:1 default) — normalize in the dispatcher.
- Legacy untagged images: shown as cosmetic fallback only, never counted in stats.

## Implementation notes

- Work happens in the session worktree `/tmp/wt-deploy-chat-db16e904` (branch `sess/db16e904`); commits there, PR at the end. No edits to `main`.
- Touch: `dashboard/sales_prompt_variations.py` (new), `dashboard/sales_image_models.py` (new), `dashboard/sales_images.py`, `dashboard/sales_image_prompts.py`, `app.py` (Phase-3 branch + backfill route + flag), `static/begin-product.html`, tests under the existing sales-test module.
