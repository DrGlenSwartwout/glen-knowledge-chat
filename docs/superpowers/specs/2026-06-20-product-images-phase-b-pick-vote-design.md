# Product Page Images — Phase B: Pick-Your-Favorite Vote Signal

**Date:** 2026-06-20
**Status:** Draft for review
**Feature flag:** `SALES_PAGES_IMAGE_VOTE` (new, ships dark)

## Context

Phase B of the 3-phase product-image effort. Phase A (merged) shows 4 images per
type (botanical + mechanism), each tagged with `prompt_variant_id` + `model_id`.
Phase B adds the **success signal**: a visitor taps a heart on their favorite
image; the vote is recorded **and attributed to that image's prompt variation and
model**, so Phase C can aggregate per-variation and per-model across all products.

- **Phase A (done):** variation- & model-tagged gallery.
- **Phase B (this spec):** heart-pick on the grid → vote captured with variation+model.
- **Phase C (later):** cross-product leaderboard + champion-challenger evolution + the impression strategy.

## Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Incentive | **Frictionless, no reward.** One tap, no points, no abuse surface. |
| Who votes | Anonymous (session cookie `amg_session`) + logged-in (email) — existing pattern. |
| Affordance | **Heart overlay** on each tile; tap to choose; chosen heart fills + tile highlight ring; one pick per type; tap another to change. Microcopy "tap your favorite". |
| Attribution | **Denormalize `prompt_variant_id` + `model_id` onto the vote at pick time** (captures what the visitor actually saw; survives later regeneration). |
| Flag | New `SALES_PAGES_IMAGE_VOTE`, independent of the pairwise `SALES_PAGES_IMAGE_PICK`. Go-live needs AI_IMAGES + IMAGE_VARIATIONS + IMAGE_VOTE all on. |
| Scope boundary | Phase B captures **picks only**. Fair pick-*rate* needs impressions → **deferred to Phase C**. |

## What already exists (reuse)

- `dashboard/sales_votes.py`: table `sales_page_votes(id, product_slug, kind, chosen_variant, session_id, email, created_at, updated_at)` with `UNIQUE(session_id, product_slug, kind)`. `record_pick(cx, slug, kind, variant, session_id, email="")` (upsert on re-pick). `get_picks(cx, slug, *, session_id, email) -> {botanical, mechanism}`. `tally`, `pair_counts` (Phase-4 helpers, untouched).
- The pairwise Phase-4 endpoint `POST /begin/product-image-pick/<slug>` (gated by `_SALES_IMAGE_PICK_ENABLED`) — left as-is. Phase A already guards the pairwise *display* branch with `and not _SALES_IMAGE_VARIATIONS_ENABLED`, so the pairwise UI and the Phase-B grid are mutually exclusive — no collision on the shared table.
- Phase A: `sales_page_images(... prompt_variant_id, model_id)`, `display_images_grouped`, the grouped data-endpoint payload, and the template `renderGrouped`.

## Architecture (Phase B)

### 1. Vote attribution columns — `dashboard/sales_votes.py`

Add nullable columns (idempotent ALTER inside `init_table`):
```
ALTER TABLE sales_page_votes ADD COLUMN prompt_variant_id INTEGER;
ALTER TABLE sales_page_votes ADD COLUMN model_id TEXT;
```
Extend `record_pick(cx, slug, kind, variant, session_id, email="", prompt_variant_id=None, model_id=None)` — the new kwargs are OPTIONAL (the Phase-4 endpoint keeps calling it without them). Persist both on insert AND on the upsert update (so a re-vote refreshes the attribution to the newly-chosen image).

### 2. Tag lookup — `dashboard/sales_images.py`

Add `tags_for(cx, slug, kind, variant) -> (prompt_variant_id, model_id)` — returns the tags of the ready image at that slot, or `(None, None)` if not found. (Small query; reused by the endpoint.)

### 3. Vote endpoint — `app.py`

New `POST /begin/product-image-vote/<slug>`:
- 404 unless `_SALES_IMAGE_VOTE_ENABLED` and `_get_product(slug)`.
- Parse JSON `{kind, variant}`; `kind` must be in `IMAGE_KINDS`; `variant` must be int ≥ 1 (no "neither" option — this is a positive favorite, not a pairwise A/B). Invalid → 400.
- `session_id = request.cookies.get("amg_session", "")`; `email` from `get_authenticated_user` (lowercased) if present.
- Look up `prompt_variant_id, model_id = sales_images.tags_for(cx, slug, kind, variant)`.
- `sales_votes.record_pick(cx, slug, kind, variant, session_id, email, prompt_variant_id=…, model_id=…)`.
- Return `{"ok": True, "picks": sales_votes.get_picks(cx, slug, session_id=…, email=…)}`.

### 4. Data endpoint — include current picks

In the Phase-A grouped branch (`app.py`, when `_SALES_IMAGE_VARIATIONS_ENABLED`), **and additionally `_SALES_IMAGE_VOTE_ENABLED`**, attach the visitor's current picks to the images body:
```python
_img_sec["body"]["picks"] = _sv.get_picks(_cx2, slug, session_id=_sess, email=_em)
```
(`_sess`/`_em` resolved from the request as the pick branch already does.) When the vote flag is off, no `picks` key (heart UI stays hidden).

### 5. Template — heart overlay — `static/begin-product.html`

In `renderGrouped`, when `body.picks` is present:
- Render a heart button overlaid on each tile (outline by default; filled + tile gets a `.sp-img-picked` ring when `body.picks[kind] === tile.variant`).
- Section microcopy: "tap your favorite".
- Click handler `vote(kind, variant)`: optimistic — immediately move the filled heart to the tapped tile, then `POST /begin/product-image-vote/<slug>` with `{kind, variant}`; on response, reconcile the hearts from `resp.picks`; on error, revert.
- One pick per kind (tapping another heart in the same kind moves it). Placeholders (still-generating tiles) get no heart.
- Add scoped CSS for the heart button + picked ring.

### 6. Feature flag — `app.py`

`_SALES_IMAGE_VOTE_ENABLED = os.environ.get("SALES_PAGES_IMAGE_VOTE", "").strip().lower() in ("1", "true", "yes")`, next to the other sales flags. OFF → no `picks` payload, no heart UI, vote endpoint 404s. Independent of `_SALES_IMAGE_PICK_ENABLED`.

## Data flow

```
visitor taps heart on a tile
  → POST /begin/product-image-vote/<slug> {kind, variant}
  → tags_for(slug, kind, variant) → (prompt_variant_id, model_id)
  → record_pick(..., prompt_variant_id, model_id)  [upsert: one vote/session/kind]
  → returns picks {botanical, mechanism}
page load (vote flag on)
  → product-page-data grouped body gains picks{} → renderGrouped fills the chosen hearts
```

## Out of scope (designed-for, not built here)

- **Phase C:** cross-product aggregation of votes by `prompt_variant_id` and `model_id`; leaderboards; champion-challenger evolution (retire/promote variations & models); console management.
- **Impressions / fair rates:** Phase B stores raw picks. Per-*variation* impressions are equal (all 4 variations shown on every product view), so variation rates compare directly. Per-*model* impressions differ (models rotate), so the model leaderboard needs an impression strategy — Phase C will reconstruct approximate model-impressions from `sales_page_images` (which models each product has) and/or add a lightweight view counter. **No impression logging in Phase B.**

## Testing (TDD, pytest, in-memory sqlite, no network; honor deploy-chat test isolation)

1. `record_pick` with tags: persists `prompt_variant_id` + `model_id`; backward-compatible 6-arg call still works (tags NULL).
2. Re-vote updates: changing the chosen variant updates `chosen_variant` AND refreshes the denormalized tags; still one row per (session, slug, kind).
3. `tags_for`: returns the right tags for a tagged image; `(None, None)` for a missing slot.
4. Endpoint logic (extract a pure helper if needed): valid `{kind, variant}` records with looked-up tags and returns `picks`; invalid kind/variant → 400; `variant` must be ≥ 1.
5. Data-endpoint picks shape: with the vote flag on, the images body carries `picks{botanical, mechanism}`; off → no `picks` key. (Verify via the helper; app can't boot in the sandbox — Pinecone-at-import; app.py edits verified via `python3 -m py_compile app.py` + the unit-tested helpers.)
6. Front-end heart interaction: manual verification (like Phase A's template task) — deferred to the human.

## Risks / notes

- **Shared table, mutually-exclusive flags:** `sales_page_votes`'s `UNIQUE(session_id, product_slug, kind)` is shared with the Phase-4 pairwise pick. They never run together (pairwise is off whenever variations/vote are on), so no overwrite in practice. Note it so it isn't accidentally broken later.
- **App import:** app.py wiring (endpoint, data-endpoint, flag) verified by `py_compile` + unit-tested helpers, not a full boot (Pinecone).
- Sandbox: use `python3` (no `python`).

## Implementation notes

- Work in the session worktree `/tmp/wt-deploy-chat-db16e904` (branch `sess/db16e904`, currently at Phase A tip `0a85ac1`, which is already merged into `main`). New Phase B commits stack here; a PR against `main` will scope to Phase B only (merge-base = `0a85ac1`).
- Touch: `dashboard/sales_votes.py` (columns + record_pick kwargs), `dashboard/sales_images.py` (`tags_for`), `app.py` (flag + vote endpoint + data-endpoint picks), `static/begin-product.html` (heart overlay + vote() + CSS), tests in a new `tests/test_sales_pages_phase_b.py`.
