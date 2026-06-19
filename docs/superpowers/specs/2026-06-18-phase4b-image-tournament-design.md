# Phase 4b — In-Funnel Sales Pages: Image Champion-Challenger Tournament

**Date:** 2026-06-18
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`)
**Builds on:** Phase 3 (images, live) + Phase 4 (votes + pick, merged). Parent: `2026-06-18-funnel-sales-pages-design.md`.

---

## Problem

Phase 4 collects pairwise votes but never acts on them. Phase 4b closes the loop: each product's images self-improve. Per product × kind, a **champion-ladder tournament** declares the crowd's preferred image, retires the loser, renders a fresh challenger, and converges on a strong champion — bounded so Flux spend stays small.

## Scope

Per product × kind (`botanical`, `mechanism`), maintain an **active pair** = champion + challenger. A periodic evaluator: counts the current head-to-head votes; on a clear winner, retires the loser and (subject to a cadence gate) enqueues one fresh challenger render; tracks the champion's successful **defenses** and marks the pair **converged** after K, stopping generation. The pick UI shows the active pair; the hero default is the champion; a converged kind shows only the champion.

**Gate:** new flag **`SALES_PAGES_IMAGE_TOURNAMENT`** (default OFF); requires `SALES_PAGES_IMAGE_PICK` on. Ships dark.

**Out of scope:** changing how the seed images or votes are produced (Phase 3/4). No bracket/round-robin (champion-ladder only).

---

## Architecture

### `dashboard/sales_image_pairs.py` — ladder state (SQLite `chat_log.db`)
- `sales_image_pairs(product_slug, kind, champion_variant, challenger_variant, defenses INTEGER DEFAULT 0, converged INTEGER DEFAULT 0, last_render_at TEXT, updated_at TEXT, PRIMARY KEY(product_slug, kind))`.
- Functions: `init_table(cx)`, `get_pair(cx, slug, kind) -> dict|None`, `ensure_pair(cx, slug, kind, ready_variants)` (lazily create from the two lowest ready variants — champion = lower, challenger = higher; None if <2), `set_pair(cx, slug, kind, *, champion, challenger, defenses, converged, last_render_at)`.
- The `sales_image_pairs` table is the sole source of truth for the active two variants per (product, kind); "retired" = any variant not in the current pair (no extra `role` column needed).

### Vote counting (extend `dashboard/sales_votes.py`)
- `pair_counts(cx, slug, kind, a, b) -> (count_a, count_b)` — picks (`chosen_variant`) equal to `a` or `b` for this product+kind (across all sessions; "neither"/0 excluded). This is the current head-to-head.

### Challenger rendering (extend the Phase-3 worker)
- A queue/worker path that renders **one** new variant for a `(slug, kind)`: pick the next variant number (`max existing + 1`), build a single prompt for that kind with a **rotated style directive** (`dashboard/sales_image_prompts.build_one_prompt(kind, variant_index)`), call `replicate_client.generate_image`, save `…/<kind>-<n>.png`, `record_image(... role omitted/NULL)`, then the evaluator promotes it to the active challenger. Reuse the existing `_drain_sales_image_queue` infra with a per-`(slug,kind,variant)` job, OR a dedicated `sales_image_challenger_queue`. Keep all Replicate calls in the scheduler (off web workers).

### Tournament evaluator (scheduler job)
A job registered in `_start_scheduler` (interval ~daily; flag-gated, no-op when off). For each `(slug, kind)` with images and a non-converged pair:
1. `ensure_pair` (lazily init champion=v1, challenger=v2 on first run).
2. `(c, ch) = pair_counts(... champion, challenger)`; `total = c + ch`.
3. If `total >= MIN_VOTES` and `max(c,ch)/total >= MARGIN`:
   - **Champion wins:** retire challenger (role=retired); `defenses += 1`. If `defenses >= K` → `converged=1` (stop). Else, if `last_render_at` older than `N` days → enqueue ONE challenger render; on completion the new variant becomes the active challenger (set_pair).
   - **Challenger wins:** new champion = challenger; retire old champion; `defenses = 0`; if cadence allows, enqueue a fresh challenger.
4. Cadence: never enqueue more than one render per pair per `N` days (`last_render_at`).

### Page-data + display (extend Phase 4's pick block)
- The pick block uses the **active pair** from `sales_image_pairs` (champion + challenger) as the two `options` (instead of variants 1&2). The non-voter hero / `display_images` returns the **champion**. A **converged** kind: no `pick` for that kind — the images body shows the champion as the hero (Phase-3-style).
- Flag off → Phase-4 behavior (variants 1&2). With tournament on but a pair not yet initialized → behaves like Phase 4 until the evaluator runs.

### Thresholds (env-tunable)
`IMAGE_TOURNAMENT_MIN_VOTES=10`, `IMAGE_TOURNAMENT_MARGIN=0.65`, `IMAGE_TOURNAMENT_CONVERGE_K=3`, `IMAGE_TOURNAMENT_CADENCE_DAYS=3`. Evaluator runs daily.

---

## Data flow

1. Phase 3 seeds variants 1 & 2; Phase 4 collects picks.
2. Evaluator (daily, flag on) reads pair state + head-to-head votes → declares a winner past the significance bar → retires loser, bumps defenses, and (cadence-permitting) enqueues one challenger render.
3. The worker renders the challenger; the evaluator (next pass) sets it as the active challenger.
4. The pick UI now shows champion vs the new challenger; votes accrue on the new head-to-head.
5. After K successful defenses the pair converges; the champion is the page's image, generation stops.

## Error handling

- Evaluator wrapped so one product's failure doesn't abort the sweep; flag-off → immediate no-op.
- A challenger render failure leaves the pair unchanged (retry next cadence window).
- `ensure_pair` requires ≥2 ready variants; if a kind has <2, it's skipped (no tournament until Phase 3 produced both).
- Vote counts use the current active variants only — retired variants' old votes don't affect the new head-to-head.
- All Replicate calls stay in the scheduler (never a web request).

## Testing

- **Pairs data layer:** `ensure_pair` inits champion/challenger from the two lowest ready variants; `set_pair`/`get_pair` round-trip; defenses/converged persist.
- **Vote counting:** `pair_counts` counts only the two active variants, excludes "neither" and other variants.
- **Evaluator (pure-ish, mocked render-enqueue):** below MIN_VOTES → no change; champion wins past MARGIN → defenses+1 + challenger enqueued (cadence permitting); defenses reaches K → converged, no further enqueue; challenger wins → promoted, defenses reset; cadence gate blocks a second render within N days.
- **Challenger render mode:** with a mocked `generate_image`, renders one new variant file + record for the next variant number.
- **Page-data:** active pair surfaced as the pick options; converged kind → champion hero, no pick; flag off → Phase-4 (variants 1&2).
- Follow deploy-chat test isolation (tmp `$DATA_DIR/chat_log.db`; mock Supabase; importorskip playwright). Visual is a manual pass.

## Flags

`…ENABLED` + `…AI_COPY` + `…AI_IMAGES` + `…IMAGE_PICK` (existing) + **`SALES_PAGES_IMAGE_TOURNAMENT`** (new, default OFF) + `REPLICATE_API_TOKEN` (already needed for images). With the tournament flag off, the pick UI is exactly Phase 4.
