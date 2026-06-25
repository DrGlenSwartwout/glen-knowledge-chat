# Biofield Intake — Live Balancing Loop (SP-B) — design notes

**Date:** 2026-06-25
**Status:** Decisions captured; full spec deferred until SP-A (Import Reveal button) ships.

SP-B reframes the intake authoring tool into a live stress-balancing loop. This file
records the confirmed design decisions so they survive until SP-B gets its own full
spec → plan. It is NOT yet a buildable spec.

## Scope

1. **Master stress list per test** — union of:
   - scan stresses (the reveal patterns / pattern_labels), and
   - additional stresses captured by voice in Phase 1.
   Each stress has an active/balanced state.

2. **Matched-remedies list per test** — from the scan AI synthesis. Each remedy
   carries the set of scan stresses (pattern codes) the synthesis matched to it.
   (Derivable from the synthesis layer structure: a layer's `patterns` belong to that
   layer's remedy; union across layers if a remedy repeats.)

3. **Two-phase live (voice) session:**
   - **Phase 1 — stress capture:** parse spoken stresses only; append to the master
     stress list. No remedies in this phase.
   - **Phase 2 — balancing:** assign remedies to layers.

4. **Auto-balance:** when a remedy is assigned to balance a layer, the active stress
   list automatically clears the stresses that remedy covers; the list then shows the
   remaining stresses still to be balanced.

5. **Layer reorder:** changing a remedy/row's layer number moves it to that position.

## Confirmed decisions (2026-06-25)

- **Layer reorder semantics:** *Insert at N + renumber* — the row moves to position N
  and the others shift to stay contiguous (1,2,3,...); no gaps, no duplicate layer
  numbers. Like dragging a row to a new slot.
- **Clear scope:** when a remedy balances a layer, it clears **all** of its
  scan-associated stresses anywhere in the master list — not just the ones in the
  assigned layer.
- **Voice stresses:** only scan stresses have AI remedy associations, so only they
  auto-clear. Voice-added (Phase-1) stresses are balanced manually (assign a remedy
  and mark cleared by hand).
- **Sequencing:** SP-A (Import Reveal button) ships first; SP-B is specced and built
  afterward.

## Forward-compatibility note for SP-A

SP-A writes reveal layers as chain rows with `most_affected` = joined pattern_labels
(text only). SP-B needs structured remedy↔stress associations; it can either extend
the schema (e.g. a `biofield_auth_stress` table + a remedy→stress link) or
re-synthesize from `e4l.db`. SP-A does not block this — it just doesn't persist the
structured associations yet.

## Open questions for the SP-B spec (not yet answered)

- Master stress list storage shape (new table columns, state machine for
  active/balanced, source = scan|voice).
- How Phase 1 vs Phase 2 are presented in the UI and how the voice grammar /
  `interpret_transcript` is split (stress-only prompt vs remedy-balancing prompt).
- How a manually-cleared voice stress is recorded and undone.
- Whether the matched-remedies list is editable / addable beyond the scan matches.
