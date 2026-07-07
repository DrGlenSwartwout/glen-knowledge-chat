# Level 2 Intake Import — Design

**Date:** 2026-07-07
**Status:** Approved (Glen: full editable review form; paste text input; parse → review → import)
**Builds on:** the intake slice (#665), the intake-on-file gate escape (#674, level 1), and the existing local intake puller (`02 Skills/intake_pull.py`).

## Goal

Import a client's existing Practice Better intake into the portal as a **real** structured record (not the level-1 external stub): paste the PB export, an LLM parses it into the declarative `INTAKE_FORM` answer shape, Glen reviews/edits every field, and on approval it is written as a submitted `intake_responses` row in prod. That satisfies the consult gate with real data, shows the actual intake in the console panel, and the existing puller tags it into `e4l.db`.

## Why the architecture is simple

The import writes a portal row whose answers use the exact `INTAKE_FORM` field ids (five Five-Fold scale fields `terrain/penetration/tissue_layer/response/commitment`, plus `health_concerns`/`obstacles`/`sleep`/`dental`/`vaccinations`, etc.). The **existing `intake_pull.py` already ingests submitted rows** by exactly those keys → `map_intake_dimensions` (confirmed Axis-B) + `extract_freetext_tags` (suggested Axis-A). So the tagger side needs **no new code**: it rides the puller (run it manually for immediate tagging, or the Monday sweep). This is a single-write feature, not a dual-write.

## Where it runs

A new page on the **local :8011 tool** (`biofield_local_app.py`), which already imports `dashboard.*` and has the JSON-mode LLM helper `_json_complete`. PHI (the pasted export) and the parse stay local; only the finished answers POST to prod.

## Flow

1. **Paste + client** — Glen opens the new page, pastes the PB export text. The parser extracts the client email from the export's Personal Information; Glen confirms or corrects it (the email keys the portal record).
2. **Parse** — `_json_complete` with a prompt that carries the `INTAKE_FORM` schema (field ids, types, scale options) and the pasted text, returning an `answers` JSON keyed by field id. Scale fields return the selected integer; the concerns table returns row objects; free-text returns strings.
3. **Review/edit** — the page renders a **full editable form** from `intake.INTAKE_FORM` (imported directly, not fetched — `/api/intake/form` is token-gated), pre-filled with the parsed answers. Every field is editable so Glen fixes any misread before saving.
4. **Import** — POST the reviewed `{email, answers}` to a new prod endpoint. It writes a submitted `intake_responses` row with the real answers plus a provenance marker `{"_imported": "practice-better"}`.
5. **Tag** — run `intake_pull.py` (manual, for immediate) or let the Monday sweep pick up the new submitted row → tags land in `e4l.db`.

## Units

| Unit | Change |
|------|--------|
| `dashboard/intake.py` (prod) | `import_response(cx, email, answers, now, source="practice-better")` — upsert a `status='submitted'` row with `answers` (the real fields) merged with `{"_imported": source}`, `form_version`, `submitted_at=now`. Same no-clobber guard as `mark_on_file`: never overwrite a real non-imported, non-external submission. |
| `app.py` (prod) | `POST /api/console/intake-import {email, answers}` — `_portal_console_ok()` gated (401), auth before body, 400 if no email or `answers` not a dict, then `import_response` under `_db_lock`. Returns `{ok:true}`. |
| `biofield_local_app.py` (local :8011) | New route(s): a page to paste + parse + review + import. `POST /author-intake/parse` (pasted text → parsed answers via `_json_complete`), and the page posts the reviewed answers to the prod import endpoint (through the local app's existing console-key + `PUBLIC_BASE_URL` plumbing). Imports `intake.INTAKE_FORM` to build the editable form. |
| `dashboard/intake_parse.py` (new, prod-importable pure module) | `build_parse_prompt(form, pasted_text) -> str` and `coerce_parsed(form, raw) -> dict` (validate/clean the LLM JSON against the schema: keep only known field ids, coerce scale values to ints in range, drop unknown keys, ensure table fields are lists of row dicts). Pure and unit-testable without the network. |
| `static/console-biofield-portal.html` (prod) | The intake read panel shows a small "imported from Practice Better" note when the response answers carry `_imported` (still renders the real fields, same as a normal submission). |

## Provenance — three states

- **Net-new portal submission:** answers, no marker.
- **Level-1 "on file":** `{"_external": true, "_note": ...}`, no real data.
- **Level-2 import:** real answers + `{"_imported": "practice-better"}`.

`is_submitted` is true for all three (status drives the gate). The puller ingests level-2 (real dims/concerns) and treats level-1 as a clean no-op (no dims/concerns). The `_imported`/`_external` keys are ignored by the puller's field extraction.

## The parse prompt

`build_parse_prompt` embeds: the section/field structure of `INTAKE_FORM` (each field's id, type, and — for scale fields — the integer→label options so the LLM picks the number), plus the pasted PB text, and instructs the model to return ONLY a JSON object keyed by field id, using the selected integer for scales, an array of `{concern, rating, years_since_onset}` for `health_concerns` (and the analogous row shapes for the other tables), and strings for text/textarea. Unknown or absent fields are omitted. `coerce_parsed` then hard-validates: drops keys not in the schema, clamps/int-coerces scale values (out-of-range → dropped so Glen sets them), and guarantees table fields are lists.

## Error handling

- Every prod route: console-key check before body; 400 on missing email or non-dict answers.
- LLM parse failure or non-JSON: the parse route returns an empty-answers result with an error flag; the page still renders the blank editable form so Glen can fill it manually (never a hard crash).
- `coerce_parsed` silently drops anything off-schema — a bad LLM value never reaches the portal record unreviewed (and Glen reviews regardless).
- No-clobber guard on import so a real portal submission is never overwritten by a later import.
- Local page holds PHI; it posts only to prod over the existing authed channel, never logs the pasted text.

## Testing

- `intake_parse.coerce_parsed`: drops unknown field ids; coerces/clamps scale ints (out-of-range dropped); wraps/validates table rows; passes clean answers through. `build_parse_prompt`: includes every section id and the scale options.
- `intake.import_response`: writes a submitted row with the real answers + `_imported`; the no-clobber guard leaves a real (non-imported, non-external) submission untouched; overwriting an `_external` level-1 stub with a real import is allowed.
- Route `POST /api/console/intake-import`: 401 without key; 400 on no email / non-dict answers; success sets `is_submitted` true and stores `_imported`.
- Round-trip: a parsed answer set for the Steven Fox sample, once imported, is ingested by `intake_pull.ingest_submissions` into the expected dimension tags (reuse the puller's existing test fixture shape).

## Out of scope (v1)

- PDF upload (paste text only).
- Batch import of multiple clients.
- Auto-import without review.
- Triggering the tagger automatically on import (rides the existing puller instead).

## Go-live

- Prod deploy-chat auto-deploys on merge (the endpoint + `intake.py`/`intake_parse.py` + console panel note).
- The local :8011 page ships in `biofield_local_app.py`: after merge, `cd ~/deploy-chat && git pull && launchctl kickstart -k gui/$(id -u)/com.glen.biofield-local-server`.
- To verify: paste the Steven Fox export, confirm the parse pre-fills the fields, edit one, import, confirm the console panel shows the record with the "imported" note, then run `intake_pull.py` and confirm the dimension tags land in `e4l.db`.
