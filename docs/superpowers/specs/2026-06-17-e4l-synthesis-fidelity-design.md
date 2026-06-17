# Spec: E4L synthesis fidelity + draft ergonomics

**Date:** 2026-06-17
**Status:** Approved (design) — pending implementation plan
**Related:** [[project_e4l_scan_ingestion]] · [[project_ascension_pricing_model]] · the multi-scan history feature (PR #157) + editor-load (#158) + the `biofield_corrections` log (PR #157). Real-world input: **Othon Molina**, the first live auto-draft — every gap below is something Glen hand-fixed on his draft.

---

## Goal

Make E4L auto-drafts arrive close enough to final that Glen barely edits them, and have the system **learn from his edits** so it needs fewer over time. Five fidelity gaps observed on Othon's draft, plus one learning loop, all localized to the synthesis pipeline and one editor button.

## Scope

In scope: the local synthesis (`02 Skills/e4l_synthesis.py` + `e4l-portal-import.py`), one editor button (`static/console-biofield-portal.html`), retaining pattern codes in portal content, and a learned-override store fed by the existing corrections log. Out of scope: the deployed app's dormant FMP path (the synthesis runs locally where FMP is reachable); the blur-reveal/state-machine (unchanged); access-gating/offer (still deferred).

## Architecture

All fidelity fixes live in the **local synthesis** (it runs on Glen's Mac, where FMP's Supabase is reachable via `SUPABASE_DB_URL` — the same connection the FMP loaders use). The editor change is one client-side button. The learning loop reuses the `biofield_corrections` log + the corrections read endpoint already shipped in PR #157.

## Components

### 1. Active-only catalog — `load_catalog`
Filter out `inactive` products so the AI can only ever pick available remedies. `load_catalog` already returns `[{slug,name,price_cents}]`; add a guard dropping entries where the product's `inactive` is truthy. (Othon's L4 "Connective Tissue Support" was `inactive:true` — this prevents proposing it.)

### 2. Catalog-constrained remedies — post-validate the LLM's FFs
The catalog constraint is today only a *prompt instruction* the LLM violated (it invented "Atlas Balance Formula"/"Vision Support"), and `to_portal_content` then let unresolved names "stay in remedy text." Change: after the LLM returns, **validate every FF against the active catalog** (`resolve_ff_slug`). Resolved FF names become the layer's `remedy` (joined with " + "); **unmatched names are dropped, never written into `remedy`**. If a layer has no resolved FF, `remedy` is **blank** for Glen to fill (agreed: a blank he fills beats a wrong auto-substitution — no nearest-match guessing).

### 3. FMP dosing — real protocol, never guessed
The LLM stops producing `dosing` (it has no reliable source). Instead a new `fmp_dosing(product_name)` queries `fmp_newapp.products` (connection `SUPABASE_DB_URL`) and sets each layer's `dosing = dosage + " " + dosage_freq + " " + dosage_timing` (the three real fields), matched to the layer's resolved FF by product name (case-insensitive). **No FMP match → blank.** This kills "take as directed" and fills accurate doses automatically. (If FMP is unreachable at synth time, all dosing falls back to blank — the draft still publishes; dosing is never fabricated.)

### 4. Clean schema
`remedy` = product name(s) only; `dosing` = the FMP protocol. No dosing text bleeding into `remedy` (Othon's L3 had "Liver Support taken one capsule daily with dinner" in `remedy`). `to_portal_content` enforces the split.

### 5. Editor "Sync order list from layers" button — `static/console-biofield-portal.html`
Gap: the order list is built once at synthesis from resolvable FFs and never re-syncs, so when Glen corrects a layer remedy he must re-add it to the order list (he had to for BFA + Macular Wellness). Add a **"Sync order list from layers"** button: client-side, it reads the current layer `remedy` fields, splits on " + ", resolves each name → active product slug using the catalog the editor already loads (`/api/console/biofield-portal/catalog`), and rebuilds the order list (deduped; preserves an existing qty/price override when the slug is unchanged); unresolved names are silently skipped. Glen clicks it after editing layers, before Publish. He keeps full control.

### 6. Learned match overrides — the system learns from confirmations
When Glen confirms a draft, the products he assigned should bias future syntheses for the **same stress patterns**, so recurring patterns reuse his choices and need no re-editing.
- **Retain patterns in content:** `to_portal_content` currently drops the LLM layer's `patterns` (the scan `item_code`s). Carry them through into each portal layer (`patterns: [item_code,...]`), stored in `content_json` (not necessarily displayed). This makes a confirmation directly yield `pattern → product` pairs.
- **Source = the corrections log:** PR #157's `biofield_corrections` already records the confirmed `content` per `email+scan_date`; with patterns retained, each confirmed layer is `(patterns, remedy)`. The local synthesis pulls corrections via `GET /api/console/biofield/corrections?since=` and builds an **override map** `item_code → preferred product` (most-recent confirmation wins).
- **Apply deterministically:** before assembling the draft, for any layer whose patterns have a learned preferred product, **set/override that layer's FF to the learned product** (and dosing from FMP for it). The LLM still groups patterns into layers and handles unknown patterns; learned patterns are pinned to Glen's choices. **Conflict rule:** if a single layer's patterns map to *different* learned products, the **most-recently-confirmed** mapping wins (each `item_code → product` entry carries the confirmation timestamp). Over time the override map grows and edits shrink.

## Data flow
scan → LLM groups patterns→FFs (active catalog only) → **validate FFs, drop unmatched** → **apply learned overrides** (pattern→product) → **FMP dosing per FF** → clean portal content (retaining `patterns`) → auto-drafted → Glen reviews/edits → **"Sync order list"** rebuilds the order from his final remedies → Publish → corrections logged → **override map grows** for the next scan.

## Error handling
- LLM emits a non-catalog FF → dropped; layer `remedy` blank (flagged by being empty in an `ai_draft`). Never an invented name.
- FMP unreachable / product absent → `dosing` blank, draft still publishes. Never fabricated.
- Corrections endpoint unreachable → synthesis runs without overrides (degrades to LLM-only), logs a notice.
- Sync button: unresolved layer remedy → skipped (not added to the order list); never errors.
- Inactive product somehow referenced → excluded by the active-only catalog.

## Testing
- `load_catalog` drops `inactive` entries.
- FF validation: an invented/inactive FF is dropped; resolved FFs join into `remedy`; no-resolution → blank `remedy`; assert no invented name ever reaches `remedy`.
- `fmp_dosing`: concatenates the 3 fields; blanks on no match; connection mocked (never hits live FMP in tests).
- `to_portal_content`: `remedy` = names only, `dosing` from FMP, `patterns` retained per layer.
- Override map: builds `item_code → product` from confirmed layers; most-recent wins; a layer with a learned pattern is pinned to the learned product (post-LLM, deterministic, testable with a stub LLM output).
- Sync button: JS syntax check + manual smoke (resolve names → slugs, dedupe, skip unresolved, preserve qty).
- Local-only units (synthesis, FMP, overrides) unit-tested with mocks; the importer wiring covered by a manual run on Othon's scan.

## Definition of done
Auto-drafts only ever propose **active** catalog products, with **real FMP dosing** (or blank), **clean schema** (no invented names, no dosing in `remedy`), patterns retained for learning; the editor has a **Sync order list from layers** button; and confirming a draft **records `pattern → product` preferences** that the next synthesis reuses. Re-running synthesis on Othon's 6/5 scan produces a draft needing materially fewer edits than the original. Unit tests green.
