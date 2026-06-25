# Biofield Intake — Balancing Loop B1: Stress Engine + UI

**Date:** 2026-06-25
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent:** SP-B live balancing loop (see `2026-06-25-biofield-intake-balancing-loop-NOTES.md`). This is the first buildable increment (B1). B2 two-phase voice, B3 comms/tag mining, B4 minimal-remedy consolidation build on this.

## Problem

SP-A added a button that imports an E4L reveal's layers + remedies into a local intake session as causal-chain rows. But the intake still has no model of the **stresses** a scan surfaces, no link between a remedy and the stresses it covers, and no live readout of what's been balanced vs what remains. Glen works a session by balancing stresses with remedies; he needs the tool to track that loop.

## Goal

Give each intake session a **master stress list** seeded from the E4L scan, a **remedy↔stress coverage map**, and an **auto-balancing** readout: as Glen puts remedies on causal-chain layers, the stresses those remedies cover drop out of the active list automatically. Add the chain **ordering rules** Glen wants (live layers on top, unbalanced scan layers trailing; insert-at-N reorder).

## Non-goals (deferred to later B-increments)

- Capturing stresses by voice (B2) — B1 seeds from the scan only, though the `source` column supports `voice/comm/tag`.
- Mining communications / health tags into stresses (B3).
- Minimal-remedy set-cover suggestions (B4).
- Rich narrative weaving of stresses into the report (B3). B1 only **lists** balanced/active stresses in the report view.

## Design

### Data model (chat_log.db — local only)

**New `biofield_auth_stress`** — one row per stress per test:
- `id INTEGER PRIMARY KEY AUTOINCREMENT`
- `test_id INTEGER` (the `_num(tid)` of the authoring test)
- `code TEXT` — E4L `item_code` (`''` allowed for future non-scan stresses)
- `label TEXT` — human label (full_name/name)
- `source TEXT` — `scan` (B1) | `voice` | `comm` | `tag` (later)
- `balance TEXT` — `required` | `optional`
- `manual_balanced INTEGER NOT NULL DEFAULT 0` — Glen marked it balanced by hand
- `created_at TEXT, updated_at TEXT`
- UNIQUE `(test_id, source, code)` for scan rows so re-seed upserts rather than duplicates.

**New `biofield_auth_remedy_coverage`** — the remedy↔stress map per test, derived from the scan synthesis:
- `id INTEGER PRIMARY KEY AUTOINCREMENT`
- `test_id INTEGER`
- `remedy TEXT` — lowercased remedy name
- `code TEXT` — a stress code this remedy covers
- UNIQUE `(test_id, remedy, code)`

**Modify `biofield_auth_chain`** — add `origin TEXT NOT NULL DEFAULT 'live'` (via `ALTER TABLE ... ADD COLUMN`, guarded like the existing `confirmed` migration). SP-A's import is updated to create rows with `origin='scan'`; rows Glen adds keep the default `'live'`.

### Auto-balance — derived, recompute-on-read (no stored balanced state)

A stress's balanced/active state is **computed**, never stored as a status, so it can never get stuck:

```
covered_codes(test) = ⋃ coverage[remedy_lower]  for every remedy currently on a chain row of this test
balanced(stress)    = stress.manual_balanced == 1  OR  stress.code in covered_codes(test)
active stresses     = those not balanced
```

So adding, changing, or deleting a chain row's remedy needs **no write** to the stress table — the next read of the stress list reflects it. Off-scan remedies (not in `coverage`) contribute nothing to `covered_codes`, so they auto-clear nothing; Glen toggles those by hand (`manual_balanced`). For display, `balanced_by` for a covered stress = the chain remedy(ies) whose coverage includes its code.

### Seeding (fold into Import Reveal + auto on scan)

Seeding runs (a) when the header is saved and a fresh scan exists for the client, and (b) on Import Reveal — both idempotent, so running twice is safe.

1. Run the scan synthesis once (reuse SP-A's pipeline). **Extend `synthesize_reveal_layers`** to also return each layer's `codes` (the `patterns` list build_payload already carries). From the layers build the coverage map: for each layer, `coverage[remedy_lower] ∪= set(codes)`.
2. Pull the scan's full findings via `biofield_e4l.scan_context(email, today)` (it returns `infoceuticals` + `stresses` groups with `code`/`name`).
3. Upsert one `biofield_auth_stress` row per finding:
   - `source='scan'`, `code`, `label`.
   - `balance='required'` when the code appears in the coverage map (the synthesis is balancing it); else `balance='optional'` (ER/MR + anything the synthesis didn't pick up — these characterize layers).
   - On re-seed, **preserve `manual_balanced`** (upsert by `(test_id,'scan',code)`, never reset the flag).
4. Replace the test's coverage rows with the freshly derived map (delete-then-insert for `source` scan; coverage is fully derived so a clean rebuild is correct).

### Chain ordering — two zones, auto-renumbered

Display order and the `layer` numbers shown are computed by a new `ordered_chain(cx, tid)`:

- **Unbalanced scan layer** = `origin='scan' AND confirmed=0`.
- **Top zone** = every other row (live rows + confirmed scan rows). Ordered by their stored `layer` (then `id`). These get display numbers `1..n`.
- **Bottom zone** = unbalanced scan layers, preserving their scan order, appended with display numbers `n+1..k`.
- Confirming an unbalanced scan layer (`confirm_row`) moves it into the top zone automatically (it now sorts with the live layers).

`authored_report` switches its `ORDER BY` to use this two-zone ordering and returns the computed display `layer` per row (the schedule/narrative already consume `authored_report`, so they inherit the ordering unchanged).

### Layer reorder (insert-at-N + renumber, within the top zone)

New `reorder_chain(cx, tid, rid, new_layer)` in `biofield_authoring`:
- Take the **top-zone** rows in current order, remove `rid`, reinsert it at position `new_layer` (clamped to `[1, len]`), then write back `layer = 1..n` in the new order.
- Unbalanced scan (bottom-zone) rows are untouched and continue to trail.
- Wired into the existing row-save route: when a row's `layer` field changes, call `reorder_chain` instead of a raw `layer` write, then the UI reloads.

### UI

A new **"Stress Balancing"** panel on `/author/<test_id>` (rendered in `biofield_report_html.py`, loaded/refreshed like the E4L panel):
- **Active (remaining)** list and a **Balanced** list. Each stress shows its `label`, a `required`/`optional` tag, and — when balanced — `manual` or the covering remedy name.
- A per-stress **balance / reactivate** toggle (sets `manual_balanced`), for optional, off-scan, and future voice stresses.
- Refreshes after any chain edit (the JS already reloads on row add/save/delete; the panel re-fetches `GET /author/<id>/stresses`).

The causal-chain editor shows the two zones (live/confirmed on top, an "Unbalanced from scan" group below) using the `ordered_chain` numbering.

Report view (`render_report_html`): a short **"Stresses balanced"** section listing balanced stresses (and, separately, any remaining active ones). Narrative weaving stays B3.

### Components / files

- **New `dashboard/biofield_stress.py`** (pure, no Flask):
  - `init_stress_tables(cx)`
  - `seed_from_scan(cx, tid, findings, coverage)` — upsert scan stresses (required/optional) preserving `manual_balanced`; rebuild coverage rows. Returns counts.
  - `covered_codes(cx, tid, remedy_names)` -> `set[str]`
  - `list_stresses(cx, tid, chain_remedy_names)` -> `{"active":[...], "balanced":[...]}` with derived state + `balanced_by`.
  - `set_manual_balanced(cx, tid, stress_id, value)`
- **Extend `dashboard/biofield_reveal_import.py`**: `synthesize_reveal_layers` adds `codes` per layer; new `build_coverage(layers) -> dict[str, set[str]]`. SP-A's `import_layers_to_test` sets `origin='scan'` on created rows.
- **Extend `dashboard/biofield_authoring.py`**: `origin` column migration; `add_chain_row(..., origin='live')`; `ordered_chain(cx, tid)`; `reorder_chain(cx, tid, rid, new_layer)`; `authored_report` uses `ordered_chain`.
- **`biofield_local_app.py`**: seed hook on header-save + Import Reveal; `GET /author/<id>/stresses`; `POST /author/<id>/stress/<sid>/balance`; row-save layer-change → `reorder_chain`.
- **`dashboard/biofield_report_html.py`**: stress-balancing panel + `importStress` JS refresh; two-zone chain rendering; report "Stresses balanced" section.

### Testing (TDD, offline)

Synthesis stubbed; sqlite tmp DBs.
1. **Seeding** — findings + coverage → required vs optional assignment (in-coverage = required); re-seed preserves `manual_balanced`; coverage rebuilt.
2. **Derived balance** — a chain remedy in the coverage map clears its codes; removing it reactivates them; off-scan remedy clears nothing; `manual_balanced` overrides regardless of chain.
3. **Ordering** — live + confirmed-scan on top by layer; unbalanced scan (`origin=scan, confirmed=0`) trails; confirming one promotes it; display numbers contiguous 1..k.
4. **Reorder** — insert-at-N renumbers the top zone; bottom-zone scan rows untouched.
5. **Routes** — `GET /stresses` returns active/balanced; `POST /stress/<id>/balance` toggles; header-save seeds when a (stubbed) fresh scan exists.

## Rollout

Local-only tool on Glen's Mac. No feature flag, no prod deploy. Additive: two new tables + one column + one module + new routes + one panel. SP-A's import is extended (sets `origin='scan'`, returns `codes`) but stays behavior-compatible for its existing tests.
