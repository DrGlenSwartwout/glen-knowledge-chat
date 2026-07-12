# Ranked candidate remedy pick-list per layer

**Date:** 2026-07-12
**Surface:** `:8011` biofield authoring tool (`biofield_local_app.py`, `/author/<test_id>`)
**Goal:** speed Glen's review of AI-drafted biofield reports by offering, per layer, a ranked
list of alternative remedies to swap in — and eliminate blank (`uncovered`) layers.

## Problem

`resolve_remedy_set` (`dashboard/biofield_stress.py`) computes ONE minimal set-cover of
remedies for a test's stress patterns. Layers it can't cover show as `uncovered` (blank), and
there is no way to see or pick alternatives. Review is slow and blank layers stall it.

## Decisions (locked with Glen)

- **Augment, not replace.** The current set-cover pick stays as each layer's pre-selected
  DEFAULT; a collapsed "alternatives" list is added beneath it. The accept-the-default fast
  path is unchanged.
- **Ranking: coverage-first with a learned boost.** Alternatives are ordered by how many of
  the layer's stress codes each remedy covers; remedies Glen has chosen before for this
  layer's pattern signature are boosted and tagged `used_before`.
- **Per layer**, not per individual pattern (one list per layer row).
- **Blank layers pull candidates from functional match** — the pattern's structure/function
  in `e4l_pattern_structures` mapped to remedies with that function, plus the existing
  formulation fallback — so a layer never shows nothing.
- **N = 5** candidates per layer (default + up to 4 alternatives).
- **Selection feeds the existing learning loop** — a swap writes through `save_remedy_set`
  (per-test) and the pattern-template save, exactly as manual edits do today; no new learning
  mechanism.

## Design

### 1. Ranking function (pure, unit-tested)
New `layer_candidates(cx, tid, chain_rows)` in `dashboard/biofield_stress.py`, built on
`resolve_remedy_set` + `_remedy_context`. Returns per layer:
```
{ "n": <layer#>, "codes": [<stress codes>], "default": <current pick or null>,
  "candidates": [ { "remedy": str, "covers": [codes], "coverage": int,
                    "source": "coverage"|"functional", "used_before": bool } ... up to 5 ] }
```
- **coverage candidates:** remedies in `biofield_auth_remedy_coverage` (via `_remedy_context`)
  that cover ≥1 of the layer's codes, sorted by coverage count desc.
- **learned boost:** remedies from `historical_remedies` / the pattern template for this
  layer's `_pattern_key` are lifted to the top of their coverage tier and flagged.
- **functional fallback (blank layers):** when no remedy covers the layer's codes, resolve
  each code's function via `e4l_pattern_structures` (code→structure/function) and offer
  remedies with that function from the catalog (+ `_formulation_fallback`), tagged
  `source: "functional"`.
- Cap at 5; the current default is always included and marked.

### 2. API (`biofield_local_app.py`)
- Extend `GET /author/<test_id>/suggest-remedies` response to include `layer_candidates`.
- Add `POST /author/<test_id>/layer/<int:n>/select` `{remedy}` → swaps that layer's remedy in
  the persisted set (`save_remedy_set`) and re-runs `resolve_remedy_set` so the panel reflects
  the new set. Returns the refreshed suggest payload.

### 3. UI (`static` /author template)
Each layer row gets a collapsed **"alternatives ▾"** control listing the ranked candidates
with a coverage indicator and a "used before" mark. Clicking one POSTs the select endpoint and
swaps in place. Default-collapsed so the fast path is unchanged.

### 4. Learning integration
The select endpoint reuses `save_remedy_set` + pattern-template save, so a swap is captured and
feeds `build_overrides` / future synthesis identically to a manual edit. No new tables.

## Testing
- Unit (`tests/test_biofield_layer_candidates.py`): coverage ordering; learned boost lifts a
  prior pick; functional fallback fills a blank layer; cap at 5; default always present.
- Route: `POST .../layer/<n>/select` persists and the next `suggest-remedies` reflects the swap.

## Out of scope
- Per-pattern (vs per-layer) candidates.
- Changing the portal (client-facing) rendering — this is the authoring tool only.
- New learning mechanisms — reuse the existing loop.
