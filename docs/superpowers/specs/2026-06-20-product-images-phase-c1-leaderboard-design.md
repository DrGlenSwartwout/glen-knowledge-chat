# Product Page Images — Phase C1: Impressions + Cross-Product Leaderboard

**Date:** 2026-06-20
**Status:** Draft for review
**Flag:** none new (exposure logging rides `SALES_PAGES_IMAGE_VOTE`; leaderboard is console-only)

## Context

Phase C of the product-image split-test, decomposed into three increments:
- **C1 (this spec):** measurement — log impressions, then a console leaderboard ranking prompt variations and models by pick-rate across ALL products. Read-only; the live page is unchanged.
- **C2 (later):** evolution loop — champion-challenger at the prompt & model level (retire weak, promote/introduce challengers), driven off C1's leaderboard.
- **C3 (later):** new-challenger generation (AI-authored prompt variations / new models).

Phases A (variation×model gallery, PR #198) and B (heart pick-vote → `sales_page_votes` tagged with `prompt_variant_id` + `model_id`, PR #199) are merged. C1 turns those votes into a fair, ranked leaderboard.

## Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Impressions | **Per-product distinct-session exposures.** One row per (product, session); dedups the front-end's generation-polling and reloads, and aligns the denominator with votes (also per-session). |
| Model denominator | Models rotate across slots, so a model's impressions = sum of exposures over the products that **contain** that model (join via `sales_page_images`). Variations are shown on every product, so their denominator is effectively total exposures — but computed by the **same** "products containing the tag" join for robustness as the active set evolves in C2. |
| Ranking | **Wilson lower-bound** (95%) on pick-rate, descending; rows below a min-volume threshold flagged "low volume" so a 1-vote tag can't top the chart. |
| Surface | **Console-only** read-only page `/console/image-leaderboard` (console-secret auth, like other `/console` pages). |
| Flag | **None new.** Exposure logging is gated by the existing `_SALES_IMAGE_VOTE_ENABLED` (same lifecycle as votes). Leaderboard page is console-auth only. |
| Scope | Cross-product (global) ranking, all-time. No per-product drill-down, no rolling window (C2 may add a window). |

## Current state (reuse)

- `sales_page_votes(... prompt_variant_id, model_id)` (Phase B) — the numerator. `session_id`, `chosen_variant`, one row per (session, product, kind).
- `sales_page_images(... prompt_variant_id, model_id, state)` (Phase A) — maps products → the variations/models they contain.
- `sales_prompt_variations` (id, label) + `sales_image_models` (id, label) — for human labels.
- Data endpoint grouped branch (`app.py`, `if _SALES_IMAGE_VARIATIONS_ENABLED:` then `if _SALES_IMAGE_VOTE_ENABLED:`) already resolves the session (`_vsess`) for picks — reuse it.
- Console auth helper `_portal_console_ok()` (used by `/admin/*` and `/console` routes).

## Architecture (C1)

### 1. Exposure logging — `dashboard/sales_image_exposures.py` (new)

```
sales_image_exposures(
  product_slug TEXT, session_id TEXT, created_at TEXT DEFAULT '',
  UNIQUE(product_slug, session_id)
)
```
- `record(cx, slug, session_id)` — `INSERT ... ON CONFLICT(product_slug, session_id) DO NOTHING` (no-op on repeat). Skips empty `session_id`.
- `per_product_counts(cx) -> {slug: n}` — `SELECT product_slug, COUNT(*) GROUP BY product_slug` (each row = a distinct session, by the UNIQUE).

**Wiring (app.py data endpoint):** inside the grouped branch, when `_SALES_IMAGE_VOTE_ENABLED` and the grouped payload has at least one ready image (`any(_grouped.values())`), call `sales_image_exposures.record(_cx2, slug, _vsess)`. Hot-path-safe: the insert is a no-op conflict after a session's first grid view; gated to fire only when images are actually shown (not during empty generation polls).

### 2. Leaderboard aggregation — `dashboard/sales_image_leaderboard.py` (new)

```python
def wilson_lower(pos, n, z=1.96):
    if n <= 0: return 0.0
    phat = pos / n
    denom = 1 + z*z/n
    centre = phat + z*z/(2*n)
    margin = z * ((phat*(1-phat) + z*z/(4*n)) / n) ** 0.5
    return (centre - margin) / denom
```

`leaderboard(cx, min_volume=30) -> {"variations": [row...], "models": [row...]}` where each `row = {key, label, votes, impressions, rate, wilson, low_volume, rank}`:
- **votes(tag):** `SELECT <tag_col>, COUNT(*) FROM sales_page_votes WHERE <tag_col> IS NOT NULL GROUP BY <tag_col>` (tag_col = `prompt_variant_id` or `model_id`).
- **impressions(tag):** sum of `per_product_counts` over the distinct products that contain the tag — `SELECT DISTINCT product_slug, <tag_col> FROM sales_page_images WHERE <tag_col> IS NOT NULL AND state='ready'`, group product-sets per tag, sum their exposure counts.
- **rate** = votes/impressions (0 if impressions 0); **wilson** = `wilson_lower(votes, impressions)`; **low_volume** = impressions < min_volume.
- **label** from the registry (`sales_prompt_variations.id→label`, `sales_image_models.id→label`); unknown/legacy → the raw key.
- Sort each list by `wilson` desc, assign `rank` 1..N (low-volume rows still listed but flagged).

### 3. Console leaderboard page — `app.py`

`GET /console/image-leaderboard`:
- `if not _portal_console_ok(): return ("unauthorized", 401)`.
- Build `leaderboard(cx)`; render a minimal HTML page: two ranked tables (Prompt Variations, Models) with columns Rank · Label · Pick-rate · Votes · Exposures · (low-volume badge). Plain server-rendered HTML in the route (no new template engine), matching the lightweight console style. Also accept `?format=json` to return the raw `leaderboard()` dict (handy for eyeballing / future C2).

### 4. No new public flag

Exposure logging fires only under `_SALES_IMAGE_VOTE_ENABLED` (data accrues whenever the vote feature is live). The console page is gated solely by console auth. Nothing on the public page changes.

## Data flow

```
visitor loads product images (vote flag on, grid has images)
  → data endpoint: sales_image_exposures.record(slug, session)   [dedup per session]
visitor taps heart → vote recorded with (prompt_variant_id, model_id)   [Phase B]
Glen opens /console/image-leaderboard
  → leaderboard(cx): join votes × images × exposures → Wilson-ranked variation & model tables
```

## Out of scope (designed-for, not built here)

- **C2:** acting on the leaderboard — prompt/model champion-challenger, retire/promote, rolling windows, assignment changes.
- **C3:** generating new prompt variations / adding models.
- Per-product drill-down; real-time dashboards; export.

## Testing (TDD, pytest, in-memory sqlite, no network; honor deploy-chat test isolation)

1. **Exposure dedup:** `record` twice for same (slug, session) → one row; different sessions → separate; empty session ignored; `per_product_counts` returns distinct-session counts.
2. **wilson_lower:** known cases — `(0,0)→0`; higher n at same rate ranks above lower n (e.g. 8/10 vs 80/100 → 80/100 has higher lower bound); monotonic sanity.
3. **Leaderboard votes/impressions:** seed images (which products contain which variation/model), exposures (per-product distinct sessions), and votes; assert per-variation and per-model votes, the correct impression denominators (model impressions = exposures of products containing it; variation = exposures of products containing it), rate, and that ranking is by wilson desc with `low_volume` set below min_volume.
4. **Labels:** registry labels resolved; legacy/unknown tag → raw key, not a crash.
5. Console route (`/console/image-leaderboard`) auth + render = manual / console-auth check (app can't boot in sandbox — Pinecone-at-import; verify app.py via `python3 -m py_compile app.py` + the unit-tested `leaderboard`/`record` helpers). `?format=json` shape covered by the helper test.

## Risks / notes

- **App import:** app.py edits (data-endpoint exposure call, console route) verified by `py_compile` + the unit-tested aggregation/record helpers, not a full boot.
- **Exposure write on a hot path:** mitigated by the UNIQUE no-op conflict + gating on "images shown". If write volume ever matters, a later optimization can batch or sample — not needed for C1.
- **Sandbox:** use `python3` (no `python`).

## Implementation notes

- Work in the session worktree `/tmp/wt-deploy-chat-db16e904` (branch `sess/db16e904`, at Phase B tip `d80dae4`, already merged into `main`). New C1 commits stack here; a PR against `main` scopes to C1 only (merge-base = `d80dae4`).
- Touch: `dashboard/sales_image_exposures.py` (new), `dashboard/sales_image_leaderboard.py` (new), `app.py` (exposure logging in the data endpoint + the console route), tests in a new `tests/test_sales_pages_phase_c.py`.
