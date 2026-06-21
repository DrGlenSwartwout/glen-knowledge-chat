# Canonical Remedy Meanings + Reveal Guardrail

**Date:** 2026-06-20
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Parent:** Follow-on to Begin #4a (the Biofield reveal). The reveal's per-remedy "meaning" (a short lay phrase) is generated fresh by the local matcher on every scan and is only editable per-reveal, so Glen's curation does not carry forward. This adds a canonical, slug-keyed meaning store that future reveals reuse, plus a guardrail that drops non-catalog remedies at ingest.

---

## Problem

A Biofield reveal lists matched remedies, each with a `meaning` (a 1-2 sentence lay description). Today:
1. The meaning is generated per scan by the local matcher and pushed via `/api/e4l/reveal-draft`. The console "Edit" action (`biofield_reveal.edit`) saves an edited meaning to that single reveal row only. There is no canonical store, so the next scan that matches the same remedy regenerates a different meaning and Glen's curation is lost.
2. Nothing validates that a remedy name maps to a real catalog product. A made-up or hallucinated name (e.g. "Mineral Binder") can enter a reveal; it just falls back to linking `/begin/match` with no product page.

## Goal

A canonical, slug-keyed meaning per catalog product that the ingest path applies to every new reveal, populated by (a) editing a meaning in the reveal console (auto-captured, on by default, with a per-remedy opt-out for one-time edits), (b) a dedicated console page that lists and curates all meanings, and (c) an AI pre-load that proposes a function-covering meaning per product. Plus a guardrail that auto-drops non-catalog remedies at ingest and records what was dropped.

## Scope

The meaning store, the ingest resolver/guardrail/override, the default-on "remember" toggle on the reveal edit, the `/console/remedy-meanings` page with AI-propose, and the `dropped` record on the reveal. Console-only; no new public flag (the existing reveal-approval flow is the member-facing gate).

**Out of scope:** changing the local matcher (it keeps emitting `name`/`slug`/`meaning`; the server applies canonical + guardrail on ingest); retroactively rewriting existing reveals (canonical applies only to NEW ingests); any change to the $1 trial / cart / pricing; a per-pattern or per-scan meaning model (meanings are per product/slug).

---

## Confirmed decisions (Glen, 2026-06-20)

- **Combine all three population mechanisms:** auto-capture reveal edits, with a "remember for the future" toggle that **defaults ON** (so editing once makes it canonical) but can be turned **off per remedy** for a one-time, scan-specific edit; PLUS a dedicated console page to keep, display, and curate all meanings; PLUS an AI pre-load.
- **Pre-load with meanings covering each remedy's major functions** (AI-generated from the product's name + ingredients + benefits, in Glen's warm lay voice).
- **Guardrail: auto-drop non-catalog remedies at ingest**, but **record the dropped names** so the drop is auditable in the console.
- **Key by slug** (post-guardrail every surviving remedy resolves to a catalog product).
- **Canonical applies only at ingest** to set a new reveal's initial meanings; never retroactive; a per-reveal edit always wins for that reveal.
- Console-only, OWNER/OPS, CONSOLE_SECRET-gated. No emoji, no em dashes.

---

## Architecture

### 1. Store - `dashboard/biofield_meanings.py`
Table `biofield_remedy_meanings (slug TEXT PRIMARY KEY, meaning TEXT, source TEXT, updated_by TEXT, updated_at TEXT)`. `source` in {`ai`, `glen`}. API (all wrapped, never raise into callers):
- `init_table(cx)`
- `upsert(cx, slug, meaning, by, source)` - insert or update.
- `get_map(cx) -> {slug: meaning}` - for the ingest override.
- `get_all(cx) -> [{slug, meaning, source, updated_at}]` - for the console page.
- `delete(cx, slug)`

### 2. Catalog resolver + guardrail + override at ingest (`app.py`, `/api/e4l/reveal-draft` ~10151)
A helper `_resolve_remedy_slug(r) -> str|None`: returns the remedy's slug if `_get_product(slug)` resolves; else looks the `name` up in `_TITLE_TO_SLUG` (exact, then case-insensitive); else `None`. In the ingest handler, before `_br.upsert`:
- For each pushed remedy, resolve a slug. Remedies that resolve to `None` are **dropped**; collect their names into `dropped`.
- Surviving remedies get `slug` set to the resolved slug.
- Load `canonical = biofield_meanings.get_map(cx)`; for each surviving remedy, if `canonical.get(slug)` is non-empty, replace `meaning` with the canonical value (override the matcher's text).
- Store the cleaned remedies via `_br.upsert(...)` as today, and persist `dropped` (see the new column). All wrapped: any failure in resolve/override falls back to keeping the pushed remedy/meaning unchanged.

### 3. `dropped` column on `biofield_reveals` (`dashboard/biofield_reveals.py`)
Add a `dropped TEXT` column (JSON list of dropped names; default `"[]"`) via an idempotent `ALTER TABLE` in `init_table` (mirrors the existing additive-column pattern). `set_dropped(cx, rid, names)`; include `dropped` in `_row` / `list_pending` so the console can show it.

### 4. Default-on "remember" promotion on the reveal edit (`dashboard/biofield_reveal_actions.py`)
`_exec_edit` already writes the edited interpretation + remedies to the reveal row. Extend it: the params carry, per remedy, a `remember` flag (the console sends it; default true). For each edited remedy whose `remember` is truthy and whose resolved slug is non-empty, `biofield_meanings.upsert(cx, slug, meaning, by, source="glen")`. The per-reveal save is unchanged; "remember" is additive. Wrapped so a meanings-store failure never fails the edit.

### 5. Console meanings page (`/console/remedy-meanings` + actions)
- `GET /console/remedy-meanings` (CONSOLE_SECRET-gated) serves `static/console-remedy-meanings.html`.
- `GET /api/console/remedy-meanings` (CONSOLE_SECRET-gated) returns `{rows: [{slug, name, meaning, source, updated_at}]}` - one row per catalog product (join the product catalog with the store; products without a meaning show empty).
- Dispatch-spine actions (`dashboard/remedy_meaning_actions.py`, OWNER/OPS, LOW_WRITE): `remedy_meaning.save` (upsert one slug, source `glen`), `remedy_meaning.delete` (delete one slug), `remedy_meaning.propose` (AI-propose one slug, store source `ai`), `remedy_meaning.propose_all` (propose for every product missing a meaning; capped + logged).
- The page lists products with an editable meaning field, a source badge, Save / Delete / "Propose with AI" per row, and "Propose all missing". XSS-safe (textContent/setAttribute).

### 6. AI propose - `propose_meaning(product) -> str` (in the meanings module or a sibling `biofield_meanings_copy.py`)
Build a prompt from the product's `name` + `ingredients` + `benefits` + `description` asking for a 1-2 sentence meaning that LEADS with the remedy's major functions, warm lay language, Glen's voice, no disease claims. Call the existing LLM helper (the one used by the ingredient-page / sales-copy generation). Returns the text, or `""` on any failure (never raises). `propose_all` iterates products with no canonical meaning, calls `propose_meaning`, upserts source `ai`, and logs the count proposed/failed.

### Reuse / untouched
- `_get_product`, `_TITLE_TO_SLUG`, the LLM helper used by ingredient/sales copy, the dispatch spine (`register_action` / `Action` / `LOW_WRITE` / OWNER/OPS), CONSOLE_SECRET gating, `biofield_reveals` store + the `biofield_reveal.edit/approve` actions, the reveal render/approval flow.
- Untouched: the local matcher, the member-facing reveal/$1-trial/cart, pricing.

---

## Data flow
1. **Pre-load:** open `/console/remedy-meanings` -> "Propose all missing" -> AI fills canonical meanings (source ai) -> Glen reviews/edits/saves the ones he wants to perfect (source glen).
2. **Ingest:** matcher pushes a draft -> guardrail drops non-catalog remedies (records names in `dropped`) -> surviving remedies get a valid slug and their meaning overridden by `canonical[slug]` if present -> cleaned remedies + `dropped` stored as a pending reveal.
3. **Reveal review:** Glen opens the reveal -> sees canonical-applied meanings + the "dropped: ..." notice + a pre-checked "remember" box per remedy -> edits a meaning -> Save (promotes to canonical unless he unchecked remember) -> Approve.
4. Member sees only the approved reveal.

## Error handling
- `_resolve_remedy_slug`, the canonical override, and all store ops are wrapped: any failure falls back to the pushed remedy/meaning unchanged; ingest never 500s on this logic.
- The guardrail NEVER rejects a push; a fully-dropped reveal still stores its interpretation (empty remedies) and the full `dropped` list. Auto-drop is always recorded.
- `propose_meaning` never raises (-> "" / skip); `propose_all` is capped and logs proposed/failed counts (no silent truncation).
- The default-on "remember" promotion is wrapped: a meanings-store failure never fails the reveal edit.
- All console endpoints/actions are CONSOLE_SECRET-gated and OWNER/OPS.

## Testing
`tests/test_biofield_meanings.py` (+ additions to the reveal/ingest tests):
- **Store:** `upsert` then `get_map`/`get_all` return it; `delete` removes it; `upsert` on an existing slug updates (single row).
- **Resolver:** a remedy with a valid `slug` resolves to that slug; a remedy with only a catalog `name` resolves via `_TITLE_TO_SLUG` (case-insensitive); a junk name -> `None`.
- **Ingest guardrail + override:** a pushed draft with a non-catalog remedy -> that remedy is absent from the stored reveal and its name is in `dropped`; a catalog remedy with a canonical meaning -> the stored meaning is the canonical one (not the pushed text); a catalog remedy with no canonical meaning -> the pushed meaning is kept; an all-non-catalog push -> reveal stored with empty remedies + full `dropped`, push still returns ok.
- **Edit promote:** `_exec_edit` with `remember=true` on a remedy -> `biofield_remedy_meanings` has that slug's meaning (source glen) AND the reveal row is updated; `remember=false` -> the reveal row is updated but the canonical store is unchanged.
- **Propose:** `propose_meaning(product)` with a mocked LLM returns the text and never raises; with the LLM raising -> returns "".
- **Console:** `/api/console/remedy-meanings` lists products joined with the store (auth-gated; 401 without the key); `remedy_meaning.save`/`.delete` mutate the store; `remedy_meaning.propose` stores an AI meaning (mocked LLM). Page serves.
- LLM mocked throughout; tmp `LOG_DB`; no emoji, no em dashes. The page interactions = manual visual pass.

## Notes
- **Console-only, no new public flag.** Canonical meanings only set a NEW reveal's initial meanings, and every reveal is still reviewed/approved before a member sees it - so even an AI-pre-loaded meaning is never shown unreviewed.
- Pre-load is AI-proposed-then-curated; the `source` badge distinguishes AI drafts from Glen-confirmed meanings on the page.
- Ties to [[reference_biofield_causal_chain_skill]] (the reveal narrative) and the #4a reveal flow.
