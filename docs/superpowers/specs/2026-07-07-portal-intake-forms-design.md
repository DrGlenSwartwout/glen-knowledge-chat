# Portal Intake Forms — Design (PB → illtowell unification slice)

**Date:** 2026-07-07
**Status:** Design approved (shape + form content + async-tagging bridge confirmed with Glen)
**Project:** PB → illtowell unification — bring Practice Better's client intake home to the portal.

## Goal

Let a client fill Dr. Glen's clinical intake **in the portal** instead of bouncing to Practice Better, as a required step before booking a Biofield Consult. On submission the answers are stored in the portal and, via a local bridge, flow into the existing clinical-tagging system with no manual export.

## Why now / context

- The Biofield Consult flow shipped with intake explicitly deferred ("C = intake form, migrate from PB"). This is that piece.
- The **downstream consumer already exists**: `~/AI-Training/02 Skills/clinical_tagger.py` `map_intake_dimensions()` (Five-Fold 1-5 answers → confirmed Axis-B tags) and `derive_intake_file_tags()` (free-text concerns → suggested Axis-A tags). Today they read an exported PB *file*; this slice makes the portal the source.
- Reuses two established patterns: the **EVOX readiness gate** (a precondition that unlocks the booking slot-picker) and the **consult portal card**.

## Scope

**In scope (this slice):**
1. A declarative intake form definition (one canonical form, versioned in code — no admin form-builder; matches the "declarative catalog, no admin UI — YAGNI" pattern used for session types).
2. A generic portal renderer for that form, including a **repeating-table** field primitive (six sections use it).
3. Draft autosave + resume (the form is long).
4. Submit → store response as JSON in the prod portal DB.
5. Gate: the consult slot-picker (`availability`/`book`) is blocked until intake is submitted ("required before booking").
6. A console read panel so Glen reads the submitted intake before the consult.
7. The **local tagging bridge**: a puller that reads new prod submissions and runs the existing tagger into `e4l.db`.

**Out of scope / deferred:**
- Multiple forms, conditional/branching logic, an admin form-builder UI.
- Intake for any entry point other than the consult (member self-serve / invite-driven were considered and set aside).
- File-upload for the portrait photo (v1 uses a text link/note field; reuse `/portal-asset/upload` later if wanted).
- Edit-after-submit by the client (v1: submit is final; Glen can request a re-fill by clearing the response console-side).

## Architecture

Two boundaries matter:

- **Prod (Render, deploy-chat):** captures + stores the intake, gates the consult, exposes submissions to the console and to the local puller. Data lives in the prod portal sqlite DB (same place as `evox_bookings`).
- **Local (Mac, vault):** the clinical tagger and `e4l.db` ledger are local-only (identical to the E4L scan split). Tagging therefore runs **asynchronously local-side**, pulling submissions from prod — the same prod→local bridge shape as E4L reveal-push / console scan-pull.

### Units

| Unit | Responsibility |
|------|----------------|
| `dashboard/intake.py` (new, prod) | The declarative `INTAKE_FORM` definition + pure logic: `validate_response`, `is_submitted`, `save_draft`, `submit`, `get_response`. Lazy-creates `intake_responses`. No Flask, no network. |
| `intake_responses` table (new, prod sqlite) | One row per client: `email` (PK), `form_version`, `status` ('draft'|'submitted'), `answers_json`, `created_at`, `submitted_at`. Lazy `CREATE TABLE IF NOT EXISTS`. |
| Routes in `app.py` (prod) | `GET /api/intake/form` (definition), `GET /api/intake/state` (draft/submitted + saved answers), `POST /api/intake/save-draft`, `POST /api/intake/submit` — all portal-token gated via `_evox_ident` (same identity helper the consult/onboarding cards use). |
| Consult gate (prod) | `consult_availability` + `consult_book` add an `intake.is_submitted(cx, email)` precondition → 409 `intake_required` when unmet. `consult_state` reports `intake_submitted: bool` so the card can show the intake step first. |
| Portal card (prod) | Intake section in the consult card (`client-portal.html`) that renders the form generically from `/api/intake/form`, autosaves drafts, and on submit reveals the existing slot-picker. |
| Console panel (prod) | Read-only rendering of a client's submitted intake in `console-biofield-portal.html`, behind `_portal_console_ok` (same gate as consult-ready). New `GET /api/console/intake/<email>`. |
| Local puller (new, vault: `02 Skills/intake_pull.py`) | Fetches submitted intakes from a console-gated prod endpoint, maps them through `map_intake_dimensions` + `derive_intake_file_tags` into `e4l.db`. Idempotent (skip already-ingested `email+form_version+submitted_at`). Wired into the weekly clinical-tags sweep. |

## The declarative form model

A field is a dict: `{id, type, label, help?, required?, options?, columns?, maps_to?}`.

**Field types:** `text`, `textarea`, `email`, `tel`, `number`, `date`, `single_choice` (radio, uses `options`), `scale` (uses `options` as ordered `{value,label}` — 1-5 or 1-10 with the exact clinical labels), `table` (repeating rows; `columns` is a list of sub-fields), `consent` (checkbox + typed signature + date, required).

`maps_to` is a hint consumed only by the local tagger: `terrain`, `penetration`, `tissue_layer`, `response`, `commitment`. Present only on the Five-Fold scale fields.

### The canonical `INTAKE_FORM` (from Glen's real PB form)

**Section 1 — Personal Information**
`first_name` text\*, `last_name` text\*, `street` text, `unit` text, `city` text, `state` text, `postal_code` text, `country` text, `email` email\*, `home_phone` tel, `mobile_phone` tel, `dob` date\*, `relationship_status` single_choice [Single, Partnered, Married, Divorced, Widowed, Prefer not to say], `gender` single_choice [Woman/Girl, Man/Boy, Nonbinary, Prefer not to say], `occupation` text, `hours_per_week` number, `referred_by` text, `favorite_color` text.

**Section 2 — Top Health Goals**
`health_concerns` table, columns: `concern` text, `rating` number (1-10), `years_since_onset` number. Help: "List your current health concerns in order of importance; rate 1-10."

**Section 3 — Key Dimensions (Clinical Theory of Everything™)** — scales with Glen's exact labels:
- `terrain` scale 1-5 `maps_to: terrain` — 1 Cancer/Degeneration/Viral/Low Energy · 2 Rapid Aging/Bacterial/Parasitic · 3 Fungal/Deposition/Slow Metabolism/Low Temp · 4 Allergy/Toxicity · 5 Stress/Hormonal.
- `penetration` scale 1-5 `maps_to: penetration` — 1 Genetic/epigenetic · 2 Cell metabolism/mitochondrial · 3 Connective tissue/immunity/autonomic/nerve · 4 Circulation/lymph · 5 Gut/dysbiosis.
- `tissue_layer` scale 1-5 `maps_to: tissue_layer` — 1 Urogenital/Muscle · 2 Connective/Immune/Cardiovascular · 3 Digestive/Respiratory · 4 Neuroendocrine · 5 Skin.
- `response` scale 1-5 `maps_to: response` — 1 No change · 2 Worse before better · 3 Mixed · 4 Gradual improvement · 5 Rapid improvement.
- `commitment` scale 1-10 `maps_to: commitment` — "1 Lowest … 10 Highest."
- `obstacles` textarea — "Anything that will get in the way of following a plan?"
- `budget_monthly` number — "Estimated $USD/month available to invest in better health."

(The 1-5 label orderings match `map_intake_dimensions` exactly, including the intentionally inverted penetration axis — verified against `clinical_dimensions.py`.)

**Section 4 — Personal Health History**
`sleep` textarea, `dental` textarea ("Any amalgams or root canals?"), `vaccinations` textarea, then tables:
- `supplements` table: `brand` text, `name` text, `reason` text, `need` number (1-10).
- `diagnoses` table: `diagnosis` text, `current` single_choice [Current, Past], `age_onset` number.
- `medications` table: `medication` text, `reason` text.
- `surgeries` table: `procedure` text, `reason` text, `age` number.
- `allergies` table: `sensitivity` text, `reaction` text.
- `portrait` textarea — "Link to a photo (or note that one was sent)."

**Section 5 — Consent**
`terms` consent\* — "I agree to the terms of service at remedymatch.com/info/terms-and-conditions." Captures checkbox + typed signature + date; required to submit.

## Data flow

1. Consult flipped ready → portal consult card renders. `consult_state` returns `intake_submitted:false` → card shows the intake form first.
2. Client fills; the page POSTs `/api/intake/save-draft` on a debounce (autosave). Response upserts the row with `status='draft'`.
3. Client submits → `POST /api/intake/submit` runs `validate_response` (required fields present, scales in range, consent signed). On pass: `status='submitted'`, `submitted_at` stamped. On fail: 400 with the list of missing/invalid field ids.
4. Card re-fetches `consult_state` → `intake_submitted:true` → slot-picker unlocks (existing consult booking UI).
5. Glen reads the intake in the console panel before the consult.
6. Local puller (weekly sweep, or on demand) pulls submitted rows → `map_intake_dimensions` writes confirmed Axis-B tags, `derive_intake_file_tags` writes suggested Axis-A tags from the free-text sections, into `e4l.db`.

## Error handling

- Every route is portal-token gated first (bad/absent token → 404 `not_found`, matching consult/onboarding), THEN body-validated (400) — token check wins over body validation (the established ordering that fixed the signal-layer Important).
- `submit` is idempotent: a second submit of an already-submitted form returns 409 `already_submitted` (no re-stamp), mirroring `existing_onboarding`.
- The consult gate returns 409 `intake_required` (not 403) so the card can distinguish "fill intake" from "not eligible."
- Draft save never blocks on validation (partial data is expected); only `submit` validates.
- Local puller: per-submission try/except so one malformed answer set can't abort the batch; best-effort, logs and continues (same shape as the tagger sweep).
- Writes go through the app's `_db_lock`; `embed()`/network (none needed here) stays outside any lock.

## Testing

- **`dashboard/intake.py` (pure):** form-definition integrity (every field has id+type; scale options ordered; `maps_to` only on the five dimension fields); `validate_response` (missing required → error list; out-of-range scale → error; unsigned consent → error; valid → ok); draft upsert then submit transitions status; `is_submitted` reflects state; repeating-table rows validate per-column.
- **Routes (mocked identity):** token gate before body (bad token → 404 even with a bad body); save-draft persists partial; submit success flips state; double-submit → 409; get form/state shapes.
- **Consult gate:** availability/book → 409 `intake_required` when not submitted; unlock after submit; `consult_state` carries `intake_submitted`.
- **Local puller:** a fixture submission maps the five dimensions to the right Axis-B tags (assert against `map_intake_dimensions`) and the concerns text yields suggested Axis-A tags; idempotent re-run writes nothing new.

## Global constraints

- No em dashes / no ALL-CAPS / no "Hook:" in any user-facing or console copy (Glen's copy rules).
- Additive only: new table + new routes + a new section in the consult card. No change to existing booking/consult byte-paths beyond the added precondition check.
- Portal-token auth via the existing `_evox_ident`; console auth via `_portal_console_ok`. No new env vars for the prod side. The local puller reuses `CONSOLE_SECRET` to read the prod endpoint (same as scan-pull).
