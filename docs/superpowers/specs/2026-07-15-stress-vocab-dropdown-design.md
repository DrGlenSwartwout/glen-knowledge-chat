# Stress-vocab dropdown — add stresses to a test from a searchable, extensible vocabulary

**Date:** 2026-07-15
**Surface:** Biofield Intake authoring page (`/author/<test_id>`, `biofield_local_app.py` + `dashboard/biofield_report_html.py`)
**Status:** Design approved, pending implementation plan

## Problem

When Glen authors a Causal Chain, stresses are captured from the transcript. Some real
stresses never get spoken, so they are missing from the test. There is currently no way
to **type a stress and add it** — the only creator is transcript capture. Glen needs to:

1. Add a stress that is **balanced by the remedies on a specific layer** (it belongs on
   that layer but wasn't picked up from the transcript). The stress must also join the
   test's overall stress list.
2. Add a plain **active** (required, not-yet-balanced) stress to the test, unassigned.
3. Search a dropdown of known stress terms by typing letters, and — when a term is new —
   add it to the shared vocabulary so it is reusable for every future client.

## Key constraint: the vocabulary is a read-only FMP mirror

`stress_vocab()` (`dashboard/biofield_authoring.py`) returns distinct `main_stress` values
from `fmp_snap_client_active_main_stress`, a table **overwritten on every FMP snapshot
re-import**. Writing new terms back into it would lose them on the next import. So new
terms need a separate, durable store that the vocab query unions in.

## Design

### Data model

New table in `chat_log.db` (the app DB; authoring is local-only, so no prod sync):

```sql
CREATE TABLE IF NOT EXISTS custom_stress_vocab(
  term        TEXT PRIMARY KEY,          -- canonical display term
  created_at  TEXT,
  created_by  TEXT DEFAULT 'glen'
);
```

`stress_vocab(cx, q, limit)` changes from "distinct FMP terms" to "distinct FMP terms
**UNION** custom terms," deduped case-insensitively, filtered by `q`, ordered, limited.
A helper `add_custom_vocab(cx, term)` inserts (idempotent via `INSERT OR IGNORE`).

This single union is what makes a typed-in term appear in the dropdown for every future
client and survive FMP re-imports.

### Endpoint

`POST /author/<test_id>/stress/add`, JSON body `{label: str, layer?: int}`:

1. `label = resolve_stress_name(cx, raw_label)` — normalize casing / match a known term
   (same normalization the head/tail inputs already use).
2. If `label` is not already in the vocabulary (FMP snapshot **or** `custom_stress_vocab`),
   `add_custom_vocab(cx, label)` — permanent + reusable.
3. `sid = add_stress(cx, test_id, label, source="manual", balance="required")` — joins the
   test's stress list, always created as `required`. `add_stress` already dedupes by
   normalized label, so re-adding an existing stress merges (and returns its existing sid)
   rather than duplicating.
4. Branch on `layer`:
   - **`layer` present** → mark the stress **balanced on that layer** by calling the same
     cover path the drag-to-cover UI uses (`cover_stress` with that layer's remedy IDs).
     This flips it from `required` to balanced and records `balanced_by` = those remedies,
     so it renders as "balanced by [those remedies]".
   - **`layer` absent** → leave it as the `required` (active, unassigned) stress from step 3.
5. Return the refreshed stress panel HTML (same shape the existing balance/assign/cover
   routes return) so the UI re-renders in place.

Balancing is therefore always done by the step-4 cover call, never by `add_stress` itself —
keeping a single source of truth for "balanced by which remedies."

### UI (both are plain `<input list=vocab>` typeaheads)

The `vocab` datalist already exists on the author page and — once `stress_vocab` unions the
custom table — will include custom terms automatically. A `<datalist>` never restricts free
text, so typing a brand-new term and pressing Enter submits it; the backend detects novelty
and persists it.

1. **Per layer card** — a small `add balanced stress…` input at the bottom of each layer's
   stress sublist. On Enter → `POST …/stress/add {label, layer:<this layer>}` → the stress
   appears balanced under that layer.
2. **Bottom of the active-stress list** — an `add active stress…` input. On Enter →
   `POST …/stress/add {label}` (no layer) → appears in the active list, ready to
   assign/cover later.

New JS helper `addStress(label, layer)` posts and calls the existing `loadStress()` refresh.

## Testing

- `stress_vocab` union: a term in `custom_stress_vocab` appears in results; a term present
  in both FMP and custom appears once (dedup); `q` filtering still works.
- `add_custom_vocab`: idempotent; case-insensitive dedup against existing FMP terms.
- Endpoint, three behaviors:
  - `{label, layer}` → stress on test **and** balanced on that layer (covered by its remedies).
  - `{label}` → stress on test as active/unassigned.
  - novel `label` → row written to `custom_stress_vocab`.
- Persistence: simulating an FMP re-import (rewriting `fmp_snap_client_active_main_stress`)
  leaves `custom_stress_vocab` intact and the custom term still returned by `stress_vocab`.

## Out of scope

- Editing/deleting custom vocab terms (add-only for now).
- Prod sync of custom vocab (authoring app is local-only).
- Any change to transcript-based capture.

## Files touched

- `dashboard/biofield_authoring.py` — `stress_vocab` union, `add_custom_vocab`, table init.
- `biofield_local_app.py` — `POST /author/<test_id>/stress/add`.
- `dashboard/biofield_report_html.py` — two typeahead inputs + `addStress()` JS.
- Table creation wired into the app's DB init (alongside the other `init_*` calls).
