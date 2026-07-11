# FF matcher → inline LLM ranking (option B, inline) — design

**Date:** 2026-07-11
**Status:** design → build
**Builds on:** FF matches Slice 3 (#777, live behind `FF_MATCHES_ENABLED`)

## Problem

The live FF matcher (option A) is raw nearest-vector top-k over the `specific-formulations`
Pinecone namespace. In prod it returns thin, sometimes self-referential results (a test
client got one match that echoed a scan item). Glen chose to upgrade to **option B, inline**:
a retrieval-augmented LLM ranker (Claude Sonnet), keeping the live "show immediately" +
generate-once architecture and the no-dosing guarantee.

## Architecture

All inside `_make_ff_items_for(email, scan_date)` (called exactly once per scan via
`ff_match_drafts.get_or_create`). No async worker, no frontend change (the card already
renders `{name, meaning, url}`; `meaning` now carries the LLM's one-line rationale).

1. **Retrieve** a broad candidate pool: `_ff_query_specific_formulations(query_text, top_k=30)`
   where `query_text` is the scan's recommendation labels joined.
2. **Filter to a safe, sellable candidate set:** for each candidate, resolve its name to a
   slug via `_resolve_buy_slug`; drop it if unresolvable, if `_ff_auto_excluded(name)` is
   true, or if its slug is already seen (dedupe). Each surviving candidate keeps
   `{name, slug, url, meaning}` (meaning = the Pinecone snippet, used as fallback text).
3. **LLM rank/select** — `_ff_llm_rank(scan_labels, candidates)`: one `_cl.messages.create`
   call on `claude-sonnet-5`, given the scan's findings and the candidate products
   (name + short description), returns a ranked selection of up to 5 as strict JSON
   `[{"name": <exact candidate name>, "why": <one-line clinical rationale>}]`. The prompt
   also restates the do-not-recommend rules as defense-in-depth. Returns `None` on any
   exception, empty/garbage parse, or model error.
4. **Build items** from the LLM ranking, constrained to the candidate map (a name the LLM
   returns that isn't a candidate is dropped — the LLM can never introduce a product):
   `{name, slug, url, meaning=why}`, in the LLM's order, capped at 5. **No `dosing` key.**
5. **Fallback:** if `_ff_llm_rank` returns `None`/empty, fall back to the current vector path
   (`ff_matcher.generate_ff_matches`), applying the same `_ff_auto_excluded` filter. The
   card never breaks; worst case it degrades to option A.

## Clinical safety (the load-bearing part)

This auto-selects products shown to clients (labelled "pending Dr. Glen's review"), so
never-recommend products must not appear even pre-review.

- **`_ff_auto_excluded(name) -> bool`** — a pure predicate encoding the DEPRECATED PRODUCTS
  rules currently living in the chat system prompt at `app.py:1638`. Excludes (case-insensitive,
  matched so canonical replacements survive):
  - Discontinued / never purchasable: **Living Water Bottle**, **Electrolyte Mineral Manna**,
    **Dental Regen Powder**.
  - Old names consolidated into canonical: bare **"Endocrine Restore"** (keep *Endocrine
    Restore Powder*), bare **"Comfort"** (keep *Comfort Synovial Syntropy*).
  - **AllerFree** (route is *Immune Modulation*).
  - **Fungifuge** — only valid as a follow-on to a Candida Cleanse; excluded from auto-match.
  - **Bioavailability Blend** / **Bioavailability Blend Powder** — adjunct-only, never a
    standalone reveal recommendation; excluded from auto-match.
- The hard filter (step 2) is the guarantee; the prompt rules and the candidate-constraint in
  `_parse_ff_rank` are defense-in-depth. An LLM hallucination cannot introduce a product
  because the output is intersected with the (already-filtered) candidate names.
- **No dosing** ever leaves `_make_ff_items_for` — the LLM is asked for name + rationale only.

## Testable units

- **`_parse_ff_rank(text, allowed_names) -> list[{name, meaning}]`** (pure): extracts a JSON
  array tolerantly (markdown fences / surrounding prose ok), keeps only objects whose `name`
  is in `allowed_names`, maps `why`→`meaning`; returns `[]` on garbage. Never raises.
- **`_ff_auto_excluded(name) -> bool`** (pure): excludes the list above; does NOT exclude
  the canonical replacements (*Endocrine Restore Powder*, *Comfort Synovial Syntropy*,
  *Immune Modulation*) or ordinary products.
- **`_make_ff_items_for`** (integration, monkeypatch `_ff_query_specific_formulations` and
  `_ff_llm_rank`): uses the LLM ranking order + meanings when present; falls back to the
  vector path when `_ff_llm_rank` returns `None`; excluded products never appear in either
  path; no item ever carries `dosing`; every item resolves to a slug + `/begin/product/` url.

## Non-goals

- The async `e4l-scan-remedy-matcher` agent / vault worker (heavier; not needed for a
  pending-review card).
- Frontend changes (the card renders the new `meaning` unchanged).
- The `healing-oasis-records` namespace (not wired into this app; `specific-formulations`
  is the FF product source).
- Structured refactor of the do-not-recommend prose in `app.py:1638` (out of scope; we mirror
  its product list in `_ff_auto_excluded`).

## Model

`claude-sonnet-5` via the existing `_cl` Anthropic client. Verify the id with a live call
during implementation; on any model/API error the fallback path keeps the feature working.
Cost is once-per-scan (generate-once cached), so a mid-tier model is affordable.
