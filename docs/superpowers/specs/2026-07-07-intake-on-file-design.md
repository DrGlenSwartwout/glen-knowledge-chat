# Intake On File — Design (level 1)

**Date:** 2026-07-07
**Status:** Approved (Glen chose level 1)
**Problem:** The intake gate shipped today ("required before booking") assumes intake happens in the portal. Existing clients already completed intake in Practice Better, so they have no `intake_responses` row and would be wrongly blocked from booking a consult (forced to re-fill). Need a console way to mark a client's intake as already on file.

## Goal

A console control (next to "Mark consult ready") that marks a client's intake as satisfied without a redundant portal re-fill. This clears the booking gate for existing clients. It does NOT fabricate clinical answers (their tags already come from scans + PB self-report separately); it records that intake was completed out of band.

## Design

### Store (`dashboard/intake.py`)
- `mark_on_file(cx, email, now, note="Completed via Practice Better")` — upsert a `status='submitted'` row whose answers are an external marker: `{"_external": True, "_note": note}`, with `form_version`, `created_at`, `submitted_at=now`. **Guard:** if a real (non-external) submitted row already exists for the email, do not overwrite it (return unchanged) — never clobber a client's real portal submission with the marker.
- `clear_intake(cx, email)` — delete the client's `intake_responses` row (to correct a mistaken mark, or reset).
- `is_external(response)` helper (or the panel checks `answers.get("_external")`).

`is_submitted` already returns True for a `status='submitted'` row, so an on-file mark satisfies the existing consult gate with no gate changes.

### Route (`app.py`)
- `POST /api/console/intake-on-file` `{email, on_file: bool}`, gated by `_portal_console_ok()` (401 unauth, 400 if no email). `on_file` true → `mark_on_file`; false → `clear_intake`. Returns `{ok:true}`.

### Console UI (`static/console-biofield-portal.html`)
- Two buttons next to the consult-ready controls (reusing the same `#email` input): **"Intake on file"** and **"Clear intake"**, calling `markIntakeOnFile(true/false)` → the new endpoint. A status span for feedback.
- In the existing intake read panel (`renderIntakeAnswers`), when the response is external (`_external`), show a single line: "Intake on file (completed externally): {note} — {submitted date}" instead of rendering form fields.

## Edge cases
- Marking on file when a real portal submission exists: no-op (guarded), so the client's real answers are preserved.
- After an on-file mark, the client's portal `save-draft`/`submit` are already no-ops-after-submit (from the finality guard), so they won't be re-prompted. "Clear intake" is the escape hatch if the mark was wrong.
- Blank email → 400.

## Copy
- Buttons: "Intake on file", "Clear intake". Panel line: "Intake on file (completed externally)". No em dashes, no ALL-CAPS.

## Testing
- `intake.mark_on_file` writes an external submitted row and `is_submitted` becomes True; the real-submission guard prevents overwrite; `clear_intake` removes the row.
- Route: console-gated (401), 400 on no email, marks + clears, token-check-before-body.

## Out of scope (level 2, later)
- Importing the actual PB intake export into the portal record (reusing `derive_intake_file_tags`/`map_intake_dimensions`).

## Go-live
Prod deploy-chat auto-deploys on merge. After merge, mark Donna Banks (sohoe1222@gmail.com) intake on file via the new endpoint so she can proceed to consult booking.
