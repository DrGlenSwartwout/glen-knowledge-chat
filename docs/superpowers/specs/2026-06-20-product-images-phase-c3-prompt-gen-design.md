# Product Page Images — Phase C3: New-Challenger Prompt Generation

**Date:** 2026-06-20
**Status:** Draft for review
**Flag:** reuses `SALES_PAGES_IMAGE_EVOLUTION` (no new flag)

## Context

The final increment of the image split-test. A (gallery, #198) → B (vote, #199) → C1 (leaderboard, #200) → C2 (evolution loop, #201) are merged. C2's **model** axis evolves today (a seeded model bench), but its **prompt** axis is wired-but-waiting: there's no supply of new prompt variations to promote. **C3 supplies them** — an LLM authors fresh prompt variations into a review queue; Glen approves; they become C2-eligible candidates. This closes the loop: generate → vote → measure → evolve → **generate more in the winning direction**.

## Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Scope | **Prompt variations only.** Models are a fixed known universe (C2 seeded that bench); you don't "generate" a model. C3 fills the prompt bench. |
| Review gate | Generated prompts land in a new `review` state (full text visible) and are **NOT** C2-eligible until Glen **Edit/Approve**s them (→ `candidate`). Reject → `retired`. Keeps unreviewed AI text out of the live pipeline. |
| Trigger | **On-demand console button** ("Generate N for botanical/mechanism") **+ a daily scheduled top-up** that refills the review queue when a kind's bench runs low. |
| Top-up threshold | Generate when a kind's `(candidate + review)` count `< 2`. |
| LLM | Reuse the codebase copy-gen standard: `anthropic.Anthropic().messages.create(model="claude-haiku-4-5-20251001")` (env-overridable). The call is an **injectable callable** so tests mock it (no live network). |
| Flag | Reuse **`SALES_PAGES_IMAGE_EVOLUTION`** — same evolution machinery/console family; gates the (paid) LLM calls, scheduler, console UI, and routes. Dark. |

## Current state (reuse)

- `sales_prompt_variations(id, kind, label, prompt_template, state, created_at, retired_at)`; `active_variations(cx, kind)` (state='active'), `candidate_variations(cx, kind)` (state='candidate'), `set_state(cx, id, state)` (C2). Phase A `build_generation_jobs` appends the no-text rule at gen time, so stored templates carry NO `_NO_TEXT` and NO product names (seeds follow this).
- `sales_image_prompts._BODY` / `_NO_TEXT` — the scene-family rules (botanical lifestyle / mechanism cell) to ground the generator.
- C1 `sales_image_leaderboard.leaderboard(cx)` — to bias generation toward winning variations.
- LLM pattern: `import anthropic; cli = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY","")); cli.messages.create(model=_MODEL, max_tokens=…, messages=[{"role":"user","content":…}])` then join `b.text` over `resp.content` text blocks (see `dashboard/inbox_ai.py`, `dashboard/ingredient_copy.py`).
- Console gate `_sales_console_ok()`; the C2 evolution section + leaderboard page `/console/image-leaderboard`; flag `_SALES_IMAGE_EVOLUTION_ENABLED`.

## Architecture (C3)

### 1. Registry: review state + helpers — `dashboard/sales_prompt_variations.py`

- `review_variations(cx, kind)` — `state='review'`, returns `[{id, kind, label, prompt_template}]`.
- `insert_variation(cx, kind, label, prompt_template, state='review') -> int` — INSERT, return new id.

### 2. Generator — `dashboard/sales_image_prompt_gen.py` (new)

```python
_MODEL = os.environ.get("IMAGE_PROMPT_GEN_MODEL", "claude-haiku-4-5-20251001")

def generate_candidates(cx, kind, n=2, *, llm=None) -> list[dict]
```
- `llm = llm or _default_llm` where `_default_llm(prompt) -> str` builds the anthropic client + `messages.create` and joins text blocks.
- Gather context: `existing = [v["prompt_template"] for v in active+candidate+review variations of this kind]` (for distinctness); the kind's scene-family rule from `sales_image_prompts._BODY[kind]`; the top 3 leaderboard variations for this kind (labels) as "what's winning, lean this way."
- Build a prompt asking for **N new, visibly distinct** scene `prompt_template`s in the same family, **no product names, no text-in-image instructions** (that rule is appended later), returning a strict JSON array `[{"label": "...", "prompt_template": "..."}]`.
- Call `llm(prompt)`; parse JSON **robustly** (slice first `[` … last `]`, `json.loads`; on failure return `[]` — never raise).
- For each item: skip if its `prompt_template` already matches an existing one (dedupe); else `insert_variation(cx, kind, label, prompt_template, 'review')`. Return the inserted rows.

### 3. Review actions + console — `dashboard/sales_image_prompt_gen.py`

- `review_action(cx, variation_id, decision, prompt_template=None) -> dict`:
  - `approve`: if `prompt_template` provided, update the row's text first; then `set_state(id, 'candidate')`. (Now C2-eligible.)
  - `reject`: `set_state(id, 'retired')`.
  - `edit`: update `prompt_template`, stay `'review'`.
  - returns `{ok, ...}`; rejects an unknown id / bad decision.
- `review_console_html(cx) -> str` — a "Prompt candidates (review)" section: per kind, a **Generate** button (`pg('generate',{kind,n:2})`) and each `review_variations` row shown with its full `prompt_template` (escaped) + **Edit/Approve/Reject** buttons (`pg('review',{id,decision,...})`), plus a tiny inline JS `pg(op, body)` that POSTs to `/console/image-prompts/<op>` and reloads. (Approve-with-edit reads the textarea value.)

### 4. Scheduled top-up — `app.py`

`_run_prompt_topup()`: `if not _SALES_IMAGE_EVOLUTION_ENABLED: return`; for each `IMAGE_KINDS`, if `len(candidate_variations)+len(review_variations) < 2`, `generate_candidates(cx, kind, n=2)`. Wrapped in try/except. Registered `scheduler.add_job(_run_prompt_topup, "interval", hours=24, id="sales_image_prompt_topup")`. (Top-up logic lives in a unit-testable `topup(cx, *, threshold=2, generate=…)` helper in the module; the job is a thin wrapper.)

### 5. Console routes + page — `app.py`

- `POST /console/image-prompts/generate {kind, n}` → `_sales_console_ok()` first, then flag-check (400), then `generate_candidates(cx, kind, int(n))` → `jsonify({ok, count})`.
- `POST /console/image-prompts/review {id, decision, prompt_template?}` → gate + flag, then `review_action(...)` → jsonify.
- Append `sales_image_prompt_gen.review_console_html(cx)` to the leaderboard page (same place the C2 evolution section is appended), only when `_SALES_IMAGE_EVOLUTION_ENABLED`.

## Data flow

```
on-demand button / daily top-up (bench < 2)
  → generate_candidates(kind): Claude writes N distinct scene prompts (grounded in scene rules + winners, distinct from existing)
  → insert as state='review'
Glen opens /console/image-leaderboard
  → "Prompt candidates (review)": Edit / Approve (→ candidate) / Reject (→ retired)
approved prompt is now a C2 candidate → C2 trials/promotes it on confidence (PR #201)
```

## Out of scope

- Generating/adding new **models** (a manual registry insert; not creative authoring).
- Auto-approving generated prompts (the review gate is intentional).
- Image-level fine-tuning, multi-variant A/B of a single prompt's wording.

## Testing (TDD, pytest, in-memory sqlite, no network; honor deploy-chat test isolation)

1. Registry: `insert_variation` returns an id and the row is in `review_variations`; `set_state` moves review→candidate / →retired (already from C2, just exercised here).
2. `generate_candidates` with an injected fake `llm` returning canned JSON → inserts N `review` variations with the right kind/labels/templates; **robust parse** (fake returns JSON wrapped in ```code fences``` and prose → still parsed); **dedupe** (a returned template identical to an existing one is skipped); a malformed `llm` response → returns `[]`, inserts nothing, no raise.
3. `review_action`: approve → state `candidate` (and edit-then-approve updates text); reject → `retired`; edit → text updated, still `review`; bad id/decision → `{ok:False}`.
4. `topup`: generates only when `(candidate+review) < threshold`; no-op when bench is full (assert the injected generate fn is/ isn't called).
5. `review_console_html`: contains the review prompt's text, Generate, Edit, Approve, Reject (escaped).
6. Routes + scheduler: console-auth + flag, manual (app can't boot — Pinecone); verify app.py via `python3 -m py_compile app.py` + the unit-tested helpers.

## Risks / notes

- **LLM cost + key:** the real `_default_llm` needs `ANTHROPIC_API_KEY`; calls are flag-gated (no spend until `SALES_PAGES_IMAGE_EVOLUTION` is on) and human-reviewed before going live. Tests never call the network (injected `llm`).
- **JSON robustness:** the model may wrap JSON in prose/fences; the parser slices `[`…`]` and tolerates failure (returns `[]`).
- **Prompt quality** is controlled by the review gate (Edit/Approve), so an occasional weak generation is harmless.
- **App import:** app.py edits verified by `py_compile` + unit-tested helpers, not a full boot. Sandbox: `python3` (no `python`).

## Implementation notes

- Work in the session worktree `/tmp/wt-deploy-chat-db16e904` (branch `sess/db16e904`, at C2 tip `edfb8e6`, already in `main`). New C3 commits stack here; a PR scopes to C3 only (merge-base = `edfb8e6`).
- Touch: `dashboard/sales_prompt_variations.py` (review_variations + insert_variation), `dashboard/sales_image_prompt_gen.py` (new — generator + review_action + topup + console HTML), `app.py` (2 routes + scheduler job + leaderboard-page append), tests in a new `tests/test_sales_pages_phase_c3.py`.
