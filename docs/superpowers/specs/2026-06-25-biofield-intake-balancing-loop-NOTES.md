# Biofield Intake — Live Balancing Loop (SP-B) — design notes

**Date:** 2026-06-25
**Status:** Decisions captured; full spec deferred until SP-A (Import Reveal button) ships.

SP-B reframes the intake authoring tool into a live stress-balancing loop. This file
records the confirmed design decisions so they survive until SP-B gets its own full
spec → plan. It is NOT yet a buildable spec.

## Scope

1. **Master stress list per test** — union of:
   - scan stresses (the reveal patterns / pattern_labels),
   - additional stresses captured by voice in Phase 1,
   - **stresses mined from recent communications** (see item 6), and
   - **stresses from health/wellness tags** (see item 7).
   Each stress has an active/balanced state and a source
   (`scan` | `voice` | `comm` | `tag`).

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

6. **Recent-communication mining (always-on for reveals AND intake):** before/at
   reveal + intake time, ALWAYS check the client's most recent communications across
   channels — email (Gmail hub `drglenswartwout@gmail.com`), chat (`chat_log.db`),
   Practice Better, GHL, etc. — **especially anything in the last 7 days** — for
   mentions of **symptoms, issues, challenges, and goals**. Itemize each as a stress
   to balance, add it to the master stress list (source=`comm`), AND weave it into the
   generated report/reveal narrative (so the client sees their own words addressed).

7. **Health/wellness tags as stresses:** pull the client's relevant health/wellness
   tags (People hub already exposes `tags`/`conditions`/`challenges`/`goals` via
   `/api/people` — `e4l-reveal-push.py:fetch_history` already reads these) and add the
   relevant ones as stresses (source=`tag`) for balancing and reporting.

8. **Minimal-remedy consolidation:** when balancing the master stress list, prefer the
   **fewest remedies** that cover the most stresses. A remedy that clears several
   stresses is preferred over several single-stress remedies. This is a balancing
   optimization over the remedy↔stress coverage map, not just a display preference.

9. **Other / optional scan stresses (characterization):** also pull in the E4L scan's
   OTHER stress patterns — the info-only `stress`-group findings (ER = Energetic
   Rejuvenators, MR1..MR10; see `biofield_e4l.py` `_STRESS_CATEGORIES=("ER","MR")`) plus
   lower-ranked patterns beyond the top set used for layers. Add them to the master
   stress list flagged **optional** (`balance = optional`, vs the required scan/comm/tag
   stresses). They are **optional to balance for now** — their primary job is to help
   **establish and characterize the causal-chain layers**. If/when one IS balanced by a
   remedy, **list it in the report as balanced**; if left unbalanced, it still appears as
   characterizing context for its layer, not as an open to-do. So every stress carries a
   `balance` attribute: `required` | `optional`.

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
- **Recent-comms + tags mining is always-on** for both reveals and intake, with a
  7-day emphasis window; mined items appear as stresses AND in the report narrative.
- **Minimal-remedy consolidation** is a hard goal of the balancing step.

## Open questions added 2026-06-25 (for the SP-B spec)

- Which comm channels are wired now vs. need new connectors? Gmail (hub) + chat_log.db
  + GHL API are reachable today; Practice Better access path TBD. Per-channel
  last-7-days fetch + an extraction pass (LLM → symptoms/issues/challenges/goals).
- Dedup across sources (a symptom in email + a matching tag + a scan pattern should
  collapse to one stress, not three).
- How `comm`/`tag` stresses get remedy associations (they have none from the scan AI):
  match them onto the scan-matched remedies' coverage, or surface for manual balancing.
- Minimal-remedy consolidation algorithm: greedy set-cover over the remedy↔stress
  coverage map; tie-break by clinical fit / dose practicality.

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
