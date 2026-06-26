# Biofield Intake — CovExt: Coverage for Non-Scan Stresses

**Date:** 2026-06-25
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent:** SP-B live balancing loop (follow-up). Independent of the Canonical-Tagging-In-House (CTI) effort.

## Problem

The balancing loop now seeds stresses from four sources — `scan`, `voice`, `tag` (profile), `comm` — but only **scan** stresses have remedy associations (the `biofield_auth_remedy_coverage` map, built from the E4L synthesis, keyed by E4L item code). Non-scan stresses (voice/tag/comm, which carry no code) therefore:
- are **not** auto-cleared by a remedy's coverage (B1 only matches by code), and
- are **excluded** from B4's minimal-remedy set-cover (it targets scan-code stresses only).

They clear today only by manual toggle or B2 label-match (a chain row head normalizing to the stress label). Glen wants the remedy-coverage **auto-balance** and the **minimal-remedy set-cover** to reach non-scan stresses too.

## Goal

Give non-scan stresses a remedy-association source so they participate in both the auto-balance and the set-cover, reusing Glen's own historical data.

## Association source — `stress_suggestions` (historical, local)

`dashboard/biofield_authoring.py:stress_suggestions(cx, stress, limit=8)` already returns the **remedies Glen has historically used for a given stress name**, from the FMP snapshot (`fmp_snap_client_remedy` ⋈ `fmp_snap_client_causal_chain` ⋈ `fmp_snap_client_active_main_stress`, matched on `main_stress`), most-used first as `[{"remedy","count"}]`. This is local, free (no LLM), and exactly the right semantics. CovExt uses it as the coverage source for non-scan stresses. Stresses with no history simply get no historical coverage (they still clear by manual/label-match).

## Design

### Shared helper

`historical_remedies(cx, label) -> set[str]` in `dashboard/biofield_stress.py`: returns the lowercased remedy names from `stress_suggestions(cx, label)` (imported lazily from `biofield_authoring`). Empty set when no history / FMP snapshot absent (never raises).

### Auto-balance — extend `list_stresses` (B1/B2)

For a stress whose `source != 'scan'` (no code), add a **historical-coverage** path alongside the existing checks. A non-scan stress is **balanced** when:
`manual_balanced` OR (existing B2 label-match: a chain row with a remedy whose head normalizes to the stress label) OR **a current chain remedy (lowercased) is in `historical_remedies(_norm(label))`**.
`balanced_by` precedence stays: covering remedy (scan code path) → label-match remedy (B2) → **historical remedy** (new) → "manual" → "". Scan stresses are unchanged (they keep the code-coverage path). Still recompute-on-read — remove the remedy and the stress reactivates.

### Set-cover — extend `suggest_minimal_remedies` (B4)

Generalize the cover from "codes" to **cover tokens**: a scan stress's token = its E4L `code`; a non-scan stress's token = `_norm(label)`. Build one `coverage = {remedy_lower: set(tokens)}` by merging:
- the existing `biofield_auth_remedy_coverage` rows (remedy → scan codes), and
- for each active non-scan required stress, `historical_remedies(label)` → that remedy "covers" the stress's norm-label token.

Then run the existing `minimal_remedies(active_tokens, coverage)` over `active_tokens` = the union of active+required scan codes **and** non-scan norm-labels, and map tokens back to display labels (codes via the existing code→label map; non-scan tokens via a norm-label→label map). Output shape unchanged (`{"picks":[{remedy,covers:[labels]}], "uncovered":[labels]}`); non-scan stresses with no history land in `uncovered` like an uncoverable scan stress.

### What does NOT change

`minimal_remedies` (pure greedy set-cover) is unchanged — it already operates on opaque tokens. The route, panel, `add_stress`, merge, and the scan/code mechanics are unchanged. CovExt only extends the two derive functions + adds one helper. Local-only (reads `fmp_snap_*` in chat_log.db); no prod deploy.

### Components / files

- `dashboard/biofield_stress.py` — `historical_remedies(cx, label)`; extend `list_stresses` (non-scan historical-coverage path) and `suggest_minimal_remedies` (token generalization + non-scan inclusion).

### Testing (TDD, offline)

Seed `fmp_snap_client_remedy`/`_causal_chain`/`_active_main_stress` so `stress_suggestions` returns a known mapping; seed stresses of each source.
1. **historical_remedies** — returns lowercased remedy names for a known stress; empty for unknown / missing snapshot.
2. **auto-balance** — a voice/tag stress is balanced when a chain remedy is in its history; removing the remedy reactivates it; `balanced_by` shows that remedy; scan stresses unchanged; precedence (code > label-match > historical > manual) holds.
3. **set-cover** — a non-scan required stress is covered by its historical remedy and appears in `picks`; a non-scan stress with no history lands in `uncovered`; scan + non-scan are covered together in one minimal set; existing B4 scan-only tests stay green.

## Rollout

Local-only tool; no feature flag, no prod deploy. Additive to the two derive functions; scan behavior and all other components unchanged.
