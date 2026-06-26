# Biofield Intake — Balancing Loop B4: Minimal-Remedy Consolidation

**Date:** 2026-06-25
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent:** SP-B live balancing loop. Builds on B1 (stress engine + coverage map, #295), B2 (#297), B3a (#300). Final SP-B increment except B3b (real comms mining).

## Problem

Glen's requirement: "consolidate balancing with a minimal number of remedies." B1 built the remedy↔stress coverage map (`biofield_auth_remedy_coverage`) from the scan synthesis, and `list_stresses` knows which stresses are still active. But nothing tells Glen the *fewest* remedies that would cover the most still-unbalanced stresses — he has to eyeball it.

## Goal

Given a test's active, required, scan-code stresses and the coverage map, compute a minimal (greedy set-cover) set of remedies that covers the most of them, and show it as a read-only suggestion so Glen can place those remedies on the chain himself.

## Non-goals

- Auto-applying suggestions to the causal chain (suggest-only).
- Covering voice/tag/optional/ER-MR stresses (they have no remedy coverage map).
- Real recent-communication mining (B3b).
- Optimal (exact) set-cover — greedy is the intended, sufficient heuristic.

## Design

### Set-cover algorithm (pure)

New `dashboard/biofield_setcover.py`:

`minimal_remedies(active_codes, coverage) -> dict`
- `active_codes`: an iterable of stress codes to cover.
- `coverage`: `{remedy: set(codes)}` — what each candidate remedy covers.
- **Greedy:** restrict each remedy's coverage to `active_codes`; repeatedly pick the remedy covering the most still-uncovered codes; tie-break deterministically by (descending coverage count, then remedy name ascending); remove its covered codes; stop when no remedy covers any remaining code.
- Returns `{"picks": [{"remedy": str, "covers": [code, ...]}, ...], "uncovered": [code, ...]}` — `covers` is the codes that pick NEWLY covers (ordered), `uncovered` is the codes no candidate reaches (sorted). Empty `active_codes` → `{"picks": [], "uncovered": []}`.
- Pure, deterministic, no DB/Flask. Unit-testable directly.

### Store helper

`suggest_minimal_remedies(cx, tid, chain_rows) -> dict` in `dashboard/biofield_stress.py`:
- Calls `list_stresses(cx, tid, chain_rows)` and selects the **active** items that are `balance=='required'`, `source=='scan'`, and have a non-empty `code` → the `active_codes` set, plus a `code -> label` map for display.
- Builds `coverage = {remedy: set(codes)}` from `biofield_auth_remedy_coverage` for the test.
- Calls `minimal_remedies(active_codes, coverage)`, then maps every code back to its label.
- Returns `{"picks": [{"remedy": str, "covers": [label, ...]}, ...], "uncovered": [label, ...]}`.
- Passing the current `chain_rows` means already-balanced stresses are excluded (they're not active), so the suggestion targets what's still open.

### Route

`GET /author/<test_id>/suggest-remedies` → `{"picks": [...], "uncovered": [...], "html": render_suggest_panel(...)}`. Read-only; computes from the live chain + coverage on each call.

### UI

A **"Suggest minimal remedies"** button near the stress panel + a read-only panel rendered by `render_suggest_panel(data)`:
- Each pick: `<remedy> → covers <label, label, ...> (<n>)`.
- An "uncovered" line listing required scan stresses no scan remedy reaches (so Glen knows what still needs a hand-picked remedy), or nothing when all covered.
- Empty state ("No active required stresses to consolidate.") when there's nothing to suggest.
- `suggestRemedies()` JS POSTs/GETs the route and swaps the panel HTML; suggest-only, no chain writes.

### Components / files

- `dashboard/biofield_setcover.py` (new) — `minimal_remedies`.
- `dashboard/biofield_stress.py` — `suggest_minimal_remedies(cx, tid, chain_rows)`.
- `biofield_local_app.py` — `GET /author/<id>/suggest-remedies`.
- `dashboard/biofield_report_html.py` — `render_suggest_panel` + button + `suggestRemedies()` JS.

### Testing (TDD, offline)

1. **minimal_remedies** — greedy picks the max-coverage remedy first; deterministic tie-break (alphabetical) on equal coverage; codes no remedy covers land in `uncovered`; redundant remedy (fully subsumed) not picked; empty input → empty.
2. **suggest_minimal_remedies** — only active+required+scan stresses are targeted (balanced/optional/voice excluded); codes map back to labels; coverage read from the table.
3. **route** — returns picks + uncovered for a seeded test; reflects the live chain (a remedy already on the chain balances its stresses → they drop out of the suggestion).
4. **UI** — panel renders picks with coverage counts, the uncovered line, and the empty state; the button + `suggestRemedies()` are present.

## Rollout

Local-only tool. No feature flag, no prod deploy. Purely additive: one new pure module, one store helper, one read-only route, one button/panel. No existing behavior changes.
