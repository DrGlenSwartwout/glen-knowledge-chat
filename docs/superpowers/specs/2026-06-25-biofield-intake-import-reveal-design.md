# Biofield Intake — Import Reveal → Causal Chain

**Date:** 2026-06-25
**Status:** Approved (design)
**Author:** Glen + Claude

## Problem

The local Biofield Intake tool (`biofield_local_app.py`, runs on Glen's Mac at
`127.0.0.1:8011`) lets Glen author a biofield analysis as a header + editable
causal-chain rows. The intake's E4L panel already detects a client's recent E4L
scan and shows the raw findings (infoceuticals / stresses).

Separately, the **Biofield Reveal** pipeline (`e4l_synthesis` + `e4l_reveal_lib`,
used by `02 Skills/e4l-reveal-push.py`) turns an E4L scan into synthesized
**layers + remedies** — the same reading clients receive in their Begin Biofield
reveal.

Today Glen must re-enter the causal chain by hand even when a fresh reveal already
captures the layers and remedies. We want a one-click bridge: pull the reveal's
layers + remedies into the intake session as causal-chain rows he can review and
adjust.

## Goal

Add a button to each intake authoring session that — when the client has a recent
E4L scan (**less than 7 days old**) — synthesizes the reveal locally and writes its
layers + remedies in as causal-chain rows, **flagged needs-review** so Glen can see
which rows came from the machine import and confirm/edit them against his own
testing.

## Non-goals

- No prod round-trip. Synthesis runs locally, in-process; PHI stays on the Mac.
- No bulk "import all sessions" action. The button is per intake session.
- No change to the reveal algorithm, the reveal-draft endpoint, or the
  `biofield_reveals` table.
- No change to the existing E4L panel freshness detection or the live-refresh flow.

## Design

### Data source — fresh local synthesis

The import runs the **same pipeline** as `e4l-reveal-push.py`:

1. Resolve the client's latest scan from `e4l.db` (or a specific `scan_id`).
2. `e4l_synthesis.pull_patterns` → top patterns; build the `label_map`
   (`item_code` → human label).
3. `e4l_synthesis.synthesize(...)` → layers, then
   `order_layers_by_pattern_count`.
4. `e4l_synthesis.to_portal_content(...)` → greeting + per-layer remedy/formulation
   resolution (age rules, formulation map).
5. `e4l_reveal_lib.build_payload(...)` → normalized layers
   `[{n, title, summary, patterns, pattern_labels, remedy: {name} | None}]`.

This is byte-for-byte the algorithm that generates client reveals, so an imported
chain matches what a reveal push would produce for the same scan. It is always
available whenever a scan exists (no dependency on a prior push/approval), and the
local app already runs under `doppler run -p remedy-match -c prd` with the keys the
pipeline needs.

The vault modules (`e4l_synthesis`, `e4l_reveal_lib`) live in `~/AI-Training/02
Skills/`. The wrapper imports them lazily with that path inserted into `sys.path`,
mirroring how `e4l-reveal-push.py` does it — so unit tests can stub the synthesis
call and run fully offline.

### The 7-day gate

The button is **enabled only when** `biofield_e4l.scan_context(email, today)`
returns `found` with `days_ago < 7`. Otherwise it renders disabled with the reason
inline:

- no scan on file → "No E4L scan on file"
- scan found but `days_ago >= 7` → "Latest scan is N days old — refresh to import"

`scan_context` already returns `found`, `scan_date`, and `days_ago`; the import
gate uses a strict `days_ago < 7` regardless of the panel's wider display window.

### Layer → causal-chain row mapping

Each synthesized layer becomes one `biofield_auth_chain` row:

| chain column   | source                                              |
|----------------|-----------------------------------------------------|
| `layer`        | layer `n`                                            |
| `head`         | layer `title` (the layer theme)                     |
| `most_affected`| `pattern_labels` joined with `", "`                 |
| `remedy`       | `remedy.name` (empty string when no catalog remedy) |
| `dosage`/`frequency`/`timing` | `remedy_dosing(cx, remedy.name)` auto-fill (blank when remedy is empty / not in catalog) |
| `confirmed`    | **0** (needs-review flag)                            |
| `sort_seq`     | 0 (existing default; rows order by layer)            |

Rows arrive **unconfirmed**. `confirmed` is a review flag, not a hard gate:
`authored_report` returns all remedy-bearing rows regardless, and the author view
renders `confirmed=0` rows with the existing `unconf` style (see
`biofield_report_html.py:441`). So imported rows are visually marked as
machine-imported and not-yet-verified; they appear in the report/schedule like any
row, and Glen confirms or edits each one with the existing confirm controls. (This
matches today's authoring behavior — the import just sets the same flag a fresh
manually-added row could carry.)

### Existing rows — append after confirm

If the session already has chain rows, the button asks (client-side):

> "This session already has N rows — add the reveal layers anyway?"

On confirm it **appends** the reveal layers (request carries `force=1`). It never
deletes or overwrites existing rows. When the session has no rows, it imports
directly.

### Components

**New module `dashboard/biofield_reveal_import.py`** (no Flask dependency):

- `synthesize_reveal_layers(email, scan_id=None, *, e4l_db, catalog, today, synth=None)`
  → `{ "found": bool, "scan_id": int|None, "scan_date": str|None,
       "days_ago": int|None, "fresh": bool, "layers": [ {n, title, summary,
       most_affected, remedy_name} ... ] }`.
  Resolves the latest (or given) scan, runs the synthesis (the `synth` param lets
  tests inject a stub for `e4l_synthesis`), and returns mapped layer dicts plus
  scan freshness meta. `fresh` is `found and days_ago < 7`. Never raises on a
  missing scan (returns `found=False`).
- `import_layers_to_test(cx, tid, layers)` → int (rows created). Creates each
  `biofield_auth_chain` row via the authoring store with `confirmed=0`, auto-filling
  dosing through `remedy_dosing`. Pure-ish — unit-testable with a real sqlite
  connection seeded by `init_auth_tables`.

**New route in `biofield_local_app.py`:**

- `POST /author/<test_id>/e4l/import-reveal` — reads the test's client email,
  calls `synthesize_reveal_layers`; if not `fresh`, returns
  `{ok: False, reason: "..."}` (HTTP 200, button stays put). If the test already
  has rows and `force` is not set, returns `{ok: False, needs_confirm: True,
  existing: N}`. Otherwise calls `import_layers_to_test`, then returns
  `{ok: True, imported: N, html: <refreshed chain/author HTML>}`.

**UI in `dashboard/biofield_report_html.py`:**

- Add the "Import Reveal → Causal Chain" button to `render_e4l_panel` (or the
  author page near the panel), enabled/disabled per the gate, with the reason text.
- Wire the click: confirm-if-rows-exist, POST to the import route, handle
  `needs_confirm` by re-POSTing with `force=1`, then swap in the returned HTML.

### Testing (TDD)

Tests live under `tests/` and stub `e4l_synthesis` so they run offline:

1. **Mapping** — a synthesized layer set maps to chain rows with the right
   `layer`/`head`/`most_affected`/`remedy` and `confirmed=0`.
2. **7-day gate** — `fresh` is True for `days_ago` 0–6, False at 7+, False when no
   scan; the route returns `ok=False` + reason when not fresh and creates no rows.
3. **Dosing auto-fill** — a remedy present in the catalog gets dosing filled; an
   empty/unknown remedy leaves dosing blank and does not error.
4. **Append vs empty** — import into an empty test creates rows directly; import
   into a test with existing rows returns `needs_confirm` without `force`, and
   appends (existing rows preserved) with `force=1`.

## Rollout

Local-only tool on Glen's Mac — no feature flag, no prod deploy. Ships when merged
and the local app is restarted. The change is additive: a new module, one route,
one button. No existing route, table, or the reveal pipeline is modified.
