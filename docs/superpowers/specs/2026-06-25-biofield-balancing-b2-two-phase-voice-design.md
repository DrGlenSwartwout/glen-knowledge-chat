# Biofield Intake — Balancing Loop B2: Two-Phase Voice Session

**Date:** 2026-06-25
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent:** SP-B live balancing loop (`2026-06-25-biofield-intake-balancing-loop-NOTES.md`). Builds on B1 (stress engine + UI, merged PR #295) and SP-A (reveal import, PR #291).

## Problem

Today the local intake's live voice session is one pass: Glen speaks the whole causal chain ("X is the head, balanced by REMEDY") and `interpret_transcript` fills causal-chain rows. But Glen works in two distinct moves — first he surveys and **names the stresses** he finds, then he **balances** them with remedies. B1 gave him a master stress list seeded from the scan; there is no way to add the additional stresses he finds by testing, and no voice path that captures stresses separately from balancing.

## Goal

Split the live voice session into two phases:

- **Phase 1 — capture stresses:** speak the stresses found while testing; each is added to the master stress list (`source='voice'`, `balance='required'`), merged by normalized label so it never duplicates a scan or earlier voice stress.
- **Phase 2 — balance with remedies:** the existing flow (spoken "[stress] balanced by [remedy]" → causal-chain rows), unchanged.

Close the loop for voice stresses: when a Phase-2 chain row's stress matches a master-list stress by normalized label, that stress shows as balanced in the Active/Balanced panel — derived, so deleting the row reactivates it.

## Non-goals (later increments)

- Mining communications / health tags into stresses (B3).
- Minimal-remedy set-cover suggestions (B4).
- Changing the Phase-2 causal-chain interpreter grammar or the schedule/narrative.

## Design

### Phase 1 — stress capture (new)

**Interpreter.** Add `interpret_stresses(transcript, complete) -> list[str]` to `dashboard/biofield_interpret.py` (beside the existing `interpret_transcript`). It uses an injected `complete(system, user) -> str` (same pattern as the causal-chain interpreter, so tests run offline) with a stress-only system prompt: extract the distinct stress / issue names the clinician spoke (e.g. "the stress is liver congestion", "also seeing adrenal fatigue") and return strict JSON `{"stresses": [str, ...]}`. It does NOT extract remedies, layers, or doses. Returns a de-duplicated, stripped list of labels; empty transcript → `[]`.

**Route.** `POST /author/<test_id>/capture-stresses` reads the saved transcript (`get_notes`), runs `interpret_stresses` with the app's injected `interpret_complete`, and for each label calls `biofield_stress.add_voice_stress(cx, test_id, label)`. Returns `{"added": n}` (count of NEW stresses actually added after merge), plus `{"error": ...}` for an empty transcript or interpreter failure (wrapped in try/except, mirroring `author_interpret`).

**Store.** `add_voice_stress(cx, tid, label) -> bool` in `dashboard/biofield_stress.py`:
- Normalizes the label with a shared `_norm(s)` helper (lowercase, collapse whitespace, strip surrounding punctuation).
- If ANY existing stress for the test (any source) normalizes to the same value, it is a **merge → no insert** (returns False).
- Otherwise inserts a row: `source='voice'`, `code=''`, `label`=the spoken label (original casing, trimmed), `balance='required'`, `manual_balanced=0`. Returns True.
- Note the B1 UNIQUE is `(test_id, source, code)`; multiple voice rows would all have `code=''`, so the normalized-label dedup (not the DB constraint) is what prevents duplicates. Voice rows must therefore be inserted with an explicit dedup check, not relied on the UNIQUE.

### Phase 2 — balancing (existing, unchanged)

The `author_interpret` route and `interpret_transcript` are untouched. Spoken balancing still produces unconfirmed causal-chain rows (`origin='live'`, top zone).

### Auto-clear by label (derived)

`list_stresses` is extended so a voice/label stress shows balanced when a chain row balances it, staying recompute-on-read:

- Signature evolves to take the current chain rows, not just remedy names:
  `list_stresses(cx, tid, chain_rows)` where `chain_rows` is a list of `{"head": str, "remedy": str}` (built by the caller from the report layers). Remedy names for the code-coverage path are derived internally from `chain_rows`; `head`s of rows that have a non-empty remedy drive the label-match path. (B1's three derive tests are updated to pass `chain_rows` dicts instead of bare remedy-name lists — a mechanical migration; behavior for scan/code stresses is unchanged.)
- A master stress is **balanced** when:
  `manual_balanced` OR `code ∈ covered_codes` (B1, scan stresses) OR there is a chain row with a non-empty remedy whose `_norm(head)` equals `_norm(stress.label)` (label match).
- `balanced_by`: the covering remedy (code path) else the matching row's remedy (label path) else `"manual"` else `""`.
- Because both paths read the live chain, removing/renaming a chain row reactivates the stress on the next read. No stored balanced state.

### UI — one widget, phase toggle

In `dashboard/biofield_report_html.py`, the "Live session (voice)" widget gains a Phase 1 ⇄ Phase 2 toggle (two small buttons or a segmented control) above the shared record/transcript controls:

- **Phase 1 — Capture stresses:** the action button reads "Capture stresses → list"; on click it POSTs to `/author/__TID__/capture-stresses`, then calls `loadStress()` to refresh the Stress Balancing panel (no page reload). A short status line reports "Added N stresses."
- **Phase 2 — Balance:** the action button reads "Interpret → fill fields" and calls the existing `interpret()` (unchanged, reloads to show new chain rows).

The toggle only swaps which action the button runs and its label; the Deepgram record controls, the `sessText` transcript box, and `/session` save are shared by both phases. Default phase on load = Phase 1 (capture), since that's the first move; Glen can switch any time.

### Components / files

- `dashboard/biofield_interpret.py` — add `interpret_stresses(transcript, complete)` + its system prompt.
- `dashboard/biofield_stress.py` — add `_norm(s)` and `add_voice_stress(cx, tid, label)`; evolve `list_stresses(cx, tid, chain_rows)` for the label-match path.
- `biofield_local_app.py` — add `POST /author/<id>/capture-stresses`; update the `author_stresses` route to pass `chain_rows` (head + remedy) into `list_stresses`.
- `dashboard/biofield_report_html.py` — phase toggle in the live-session widget + `captureStresses()` JS.

### Testing (TDD, offline)

`complete()`/`interpret_complete` injected; sqlite tmp DBs.
1. **interpret_stresses** — a transcript with several spoken stresses → the label list; empty transcript → `[]`; remedies/doses ignored.
2. **add_voice_stress** — inserts a voice stress (required, source=voice); a normalized-duplicate of an existing scan or voice stress merges (no insert, returns False); casing/whitespace/punctuation variations dedup.
3. **list_stresses label-match** — a chain row whose head normalizes to a voice stress marks it balanced with `balanced_by`=that row's remedy; removing the row reactivates it; a row with an empty remedy does NOT balance; scan/code path unchanged.
4. **capture-stresses route** — posts run the interpreter (stubbed) and add voice stresses; empty transcript → error; returns the added count.
5. **panel/UI** — the phase toggle renders both modes; `captureStresses()` present and posts to the capture route.

## Rollout

Local-only tool on Glen's Mac. No feature flag, no prod deploy. Additive: one new interpreter function, one new store function, one evolved function (with its B1 tests migrated), one new route, one updated route, and a phase toggle. Phase 2 and the schedule/narrative are untouched.
