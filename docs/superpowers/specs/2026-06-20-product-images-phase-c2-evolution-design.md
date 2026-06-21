# Product Page Images — Phase C2: Evolution Loop (champion-challenger)

**Date:** 2026-06-20
**Status:** Draft for review
**Flag:** `SALES_PAGES_IMAGE_EVOLUTION` (new, ships dark)

## Context

Phase C2 of the image split-test. C1 (PR #200) measures — a console leaderboard ranking prompt variations and models by Wilson-lower-bound pick-rate across products. **C2 acts on it:** a human-in-the-loop champion-challenger engine that proposes retiring an underperforming active prompt/model and promoting a benched candidate; Glen approves in the console; the active set changes; new products generate with it automatically.

- C1 (done): impressions + leaderboard.
- **C2 (this spec):** evolution engine + proposals + console approve/reject + manual trial controls.
- C3 (later): new-challenger generation (AI-authored prompt variations) — fills the prompt bench.

## Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Autonomy | **Propose & approve in console** (human-in-the-loop). The engine recommends; Glen approves; nothing changes live without approval. |
| Confidence rule | A swap is proposed only when the weakest active item is **confidently worst**: its Wilson **upper** bound < the best active item's Wilson **lower** bound (intervals separated), AND its impressions ≥ `min_impressions` (default 50), AND a candidate is benched. |
| Manual controls | Console also offers **"trial candidate X"** (swap it in for the weakest active) and proposal **reject** — so you can explore on demand when the auto-trigger doesn't fire (4 similar prompts rarely separate). |
| Cadence | A **daily** scheduler recomputes proposals (`interval`, hours=24), like the existing tournament job. |
| Set-size invariant | Every change is a **swap** (retire 1 + promote 1), so the active set stays exactly 4 variations/kind (Phase A slot mapping) and a fixed model count. No standalone retire (would shrink the grid). |
| Bench | Add registry state `candidate`. **Seed model candidates** (Ideogram V3, Flux Ultra, SD 3.5) so the model axis evolves immediately. The prompt bench stays empty until C3. |
| Flag | `SALES_PAGES_IMAGE_EVOLUTION` gates the scheduler + console actions + candidate seeding. Dark. |

## Current state (reuse)

- Registries `sales_prompt_variations(id, kind, label, prompt_template, state)` + `sales_image_models(id, label, engine, engine_ref, state)`; `active_variations(cx, kind)` (state='active', ORDER BY id), `active_models(cx)` (state='active', ORDER BY rowid). Phase A generation reads these — **changing `state` changes what new products generate, automatically.**
- C1 `dashboard/sales_image_leaderboard.py`: `leaderboard(cx, min_volume)` rows `{key, label, votes, impressions, rate, wilson, low_volume, rank}`; `wilson_lower(pos, n, z)`.
- Console gate `_sales_console_ok()`; the leaderboard page `/console/image-leaderboard` (C1).
- Scheduler: `BackgroundScheduler` with `scheduler.add_job(...)` (app.py ~16810; `_run_image_tournament` daily is the model to mirror). `_SALES_*_ENABLED` flag pattern.

## Architecture (C2)

### 1. Registry: candidate state + helpers

Add to **both** registries (`dashboard/sales_prompt_variations.py`, `dashboard/sales_image_models.py`):
- `set_state(cx, id, state)` — flip an item's `state` (id is INTEGER for variations, TEXT for models).
- `candidate_variations(cx, kind)` / `candidate_models(cx)` — `state='candidate'`.
- `seed_candidates(cx)` (models only) — `INSERT OR IGNORE` the bench models as `state='candidate'`:
  - `ideogram-v3` / "Ideogram V3" / `ideogram-ai/ideogram-v3-quality`
  - `flux-ultra` / "Flux 1.1 Pro Ultra" / `black-forest-labs/flux-1.1-pro-ultra`
  - `sd-3.5-large` / "Stable Diffusion 3.5 L" / `stability-ai/stable-diffusion-3.5-large`
  - **Confirm exact Replicate refs + pricing at build** (same caveat as Phase A). Seeding runs only under the flag.

### 2. Wilson interval — `dashboard/sales_image_leaderboard.py`

Add `wilson_upper(pos, n, z=1.96)` (mirror of `wilson_lower`, `+ margin`). Used for the interval-separation test.

### 3. Evolution engine — `dashboard/sales_image_evolution.py` (new)

```
sales_image_evolution_proposals(
  id INTEGER PK, axis TEXT, kind TEXT,           -- axis 'model'|'variation'; kind '' for models, botanical/mechanism for variations
  retire_key TEXT, promote_key TEXT, stats_json TEXT,
  state TEXT DEFAULT 'pending',                  -- pending | approved | rejected
  created_at TEXT, decided_at TEXT DEFAULT '')
sales_image_evolution_log(
  id INTEGER PK, axis TEXT, kind TEXT, retired_key TEXT, promoted_key TEXT,
  actor TEXT, created_at TEXT, undone_at TEXT DEFAULT '')
```

- `propose(cx, *, min_impressions=50)` → list of proposal dicts. For each axis (models; variations per kind):
  - Pull the C1 leaderboard rows, intersect with the currently **active** items (active_models / active_variations(kind)).
  - Among active items with impressions ≥ `min_impressions`, find weakest (lowest `wilson`) and best (highest `wilson`).
  - If `wilson_upper(weakest) < wilson_lower(best)` **and** a candidate is benched → emit `{axis, kind, retire_key, retire_label, promote_key (next candidate), promote_label, stats}`.
  - Persist as a **pending** row, **deduped** (no duplicate pending for the same axis/kind/retire/promote) and on **cooldown** (skip if an identical proposal was `rejected` within the last 14 days).
- `pending_proposals(cx)` → the pending rows (with stats) for the console.
- `decide(cx, proposal_id, decision, actor)` — `decision in ('approve','reject')`. On approve → `_apply_swap(cx, axis, kind, retire_key, promote_key, actor)`; mark approved + `decided_at`. On reject → mark rejected.
- `trial(cx, axis, kind, candidate_key, actor)` — manual: pick the weakest active item for (axis, kind) as `retire_key`, swap candidate in (`_apply_swap`). (Errors if no active items or candidate isn't a candidate.)
- `_apply_swap(cx, axis, kind, retire_key, promote_key, actor)` — set retire_key `state='retired'`, promote_key `state='active'` (via the registry `set_state`), and append a `sales_image_evolution_log` row. Asserts the swap keeps the active count unchanged.
- `undo(cx, log_id, actor)` — reverse the last swap: retired_key → 'active', promoted_key → 'candidate'; set `undone_at`.

### 4. Scheduler — `app.py`

`_run_image_evolution()`: `if not _SALES_IMAGE_EVOLUTION_ENABLED: return`; `with sqlite3.connect(LOG_DB) as cx: sales_image_evolution.propose(cx)`. Register beside the others: `scheduler.add_job(_run_image_evolution, "interval", hours=24, id="sales_image_evolution")`.

### 5. Console — `app.py`

Extend the **C1 leaderboard page** (`/console/image-leaderboard`) so that, when `_SALES_IMAGE_EVOLUTION_ENABLED`, it also renders: pending proposals (the swap + supporting stats + **Approve/Reject** buttons) and, per axis/kind, the benched candidates with a **"Trial"** button + an **undo** of the last swap. Action POST routes (gated by `_sales_console_ok()`):
- `POST /console/image-evolution/decide` `{proposal_id, decision}` → `evolution.decide`.
- `POST /console/image-evolution/trial` `{axis, kind, candidate_key}` → `evolution.trial`.
- `POST /console/image-evolution/undo` `{log_id}` → `evolution.undo`.
Each returns JSON `{ok, ...}`; the page re-fetches. (Lightweight routes, consistent with C1's console route; the `dashboard/sales_pages_actions.py` action-spine is an alternative but not required for C2.)

### 6. Flag

`SALES_PAGES_IMAGE_EVOLUTION` (→ `_SALES_IMAGE_EVOLUTION_ENABLED`). Gates: the scheduler job, candidate seeding, the console proposal/candidate UI, and the action routes. OFF → no proposals, no candidates seeded, leaderboard page unchanged (C1 read-only only). Independent of the other flags (but only meaningful once `SALES_PAGES_IMAGE_VOTE` is accruing data).

## Data flow

```
daily: _run_image_evolution → evolution.propose(cx)
  → per axis/kind: weakest active confidently worst (Wilson intervals separated) + candidate benched
  → persist pending proposal (deduped, cooldown on rejected)
Glen opens /console/image-leaderboard
  → sees leaderboard (C1) + pending proposals + benched candidates
  → Approve a proposal | Trial a candidate | Undo last swap
  → _apply_swap: retire one (state=retired) + promote one (state=active) + log
new product generation (Phase A) reads active_variations/active_models → uses the evolved set
```

## Out of scope (designed-for, not built here)

- **C3:** generating new prompt variations to fill the prompt bench (C2's prompt axis is wired but inert until candidates exist).
- Multi-armed-bandit auto-exploration; per-product evolution; rolling time windows on the leaderboard; A/B holdouts.

## Testing (TDD, pytest, in-memory sqlite, no network; honor deploy-chat test isolation)

1. Registry: `set_state` flips state; `candidate_models`/`candidate_variations` filter; `seed_candidates` is `INSERT OR IGNORE` (idempotent, doesn't touch active rows).
2. `wilson_upper`: > `wilson_lower` for the same (pos,n); together they bracket the rate; n≤0 → (0,0) sane.
3. `propose`: fires a swap only when the weakest active is interval-separated below the best AND impressions ≥ min AND a candidate exists; does NOT fire on overlapping intervals, on low impressions, or with no candidate; emits the right retire/promote keys; dedup + cooldown work.
4. `_apply_swap` / `decide(approve)`: retire→'retired', promote→'active', active count unchanged, log row written. `decide(reject)` → rejected, no state change.
5. `trial`: swaps the named candidate in for the current weakest active; rejects a non-candidate key.
6. `undo`: reverses the last swap (retired→active, promoted→candidate), sets `undone_at`.
7. Console routes + scheduler: console-auth + manual (app can't boot in sandbox — Pinecone-at-import; verify app.py via `python3 -m py_compile app.py` + the unit-tested engine). Scheduler job = thin wrapper over `propose`.

## Risks / notes

- **Candidate Replicate refs/pricing** for ideogram-v3 / flux-ultra / sd-3.5-large — confirm against the account's token before enabling (seeding is flag-gated, so safe to merge dark).
- **Set-size invariant:** `_apply_swap` must keep the active count constant (assert); a swap is always retire-1 + promote-1. Prompt swaps stay within one `kind`.
- **App import:** app.py edits (scheduler, console routes/page) verified by `py_compile` + unit-tested engine, not a full boot.
- Sandbox: `python3` (no `python`).

## Implementation notes

- Work in the session worktree `/tmp/wt-deploy-chat-db16e904` (branch `sess/db16e904`, at C1 tip `753f4b9`, already in `main`). New C2 commits stack here; a PR scopes to C2 only (merge-base = `753f4b9`).
- Touch: `dashboard/sales_prompt_variations.py` + `dashboard/sales_image_models.py` (state helpers, model candidates), `dashboard/sales_image_leaderboard.py` (`wilson_upper`), `dashboard/sales_image_evolution.py` (new — engine + tables), `app.py` (flag, scheduler job, console page additions + action routes), tests in a new `tests/test_sales_pages_phase_c2.py`.
