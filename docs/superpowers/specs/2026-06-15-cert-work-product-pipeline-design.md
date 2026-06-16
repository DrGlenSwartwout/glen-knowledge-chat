# Certification Work-Product Submission Pipeline — Build Spec

**Status:** Design approved (Glen, 2026-06-15). Ready for plan → implement.
**Program design (source of truth for rules):** `~/AI-Training/00 Projects/certification/cert-work-product-framework.md`.
**Repo:** glen-knowledge-chat (deploy-chat). **Domain:** illtowell.com. **Branch:** `sess/b76661d9-cert-submissions`.

## Goal

Let certification students submit their published (or publishable) project work through a
portal we control, let Glen/Rae review it in the console behind **two gates** (approve →
publish), track each student's progress against the certification completion rules, and feed
approved-and-published work into the existing `case-studies` proof library.

## Background / why

Certification benefits are being reworked to be scalable and ~zero-marginal-cost. "Publish their
work" is the first piece. Students earn certification by producing a body of published work; we
get a stream of attributable proof for the funnel/chat. The existing third-party feedback/results
portal stays a valid channel but has limits we don't control, so we add our own.

## Completion rules (from the framework doc — the tracker enforces these)

- **≥ 12 approved submissions.**
- **All 12 module concepts covered** (one submission may cover several).
- **Multiple modalities, including written + video.** Minimum (B): at least one written and one
  video across the body of work. Encouraged (A): each entry in both written and video form.
- **Published or publishable with permission** to publish + reuse in whole or part (social
  distribution qualifies; professional publication encouraged).
- Rules subject to change; keep them in one place (`cert_rules.py`) so they're easy to adjust.

## Scope

### In scope (v1)

1. **`cert_submissions` store** (`dashboard/cert_submissions.py`) — sqlite, mirrors
   `dashboard/cert_bonus.py` conventions (LOG_DB / `chat_log.db`, `init_tables(cx)`, ISO-UTC
   `_now()`, idempotent writes). Holds each submission with its tags + status.
2. **`cert_rules.py`** (`dashboard/cert_rules.py`) — the 12 module concepts, the format list, and
   a pure `evaluate(submissions) -> dict` that returns progress against the completion rules
   (approved count, modules covered set, has_written, has_video, complete?). No I/O. Easy to edit.
3. **Student portal** — magic-link login (reuse the practitioner magic-link pattern) + a
   submission form. Page `static/cert-portal.html`; routes for login + submit + "my submissions".
4. **Console review surface** — a board page `static/console-cert.html` + console-gated APIs to
   list submissions, approve/return (Gate 1), and publish (Gate 2).
5. **Publish on Gate 2** — embed the approved piece's text + upsert into the Pinecone
   `case-studies` namespace (so it surfaces as proof via the existing `_query_proof_cards`),
   with attribution metadata (practitioner name + the published URL).
6. **modules_completed sync** — when a student's covered-module count changes (on approve), update
   `modules_completed` on their practitioner record via the existing helper path, so the count
   that already drives the cert-tiered referral stays in sync.
7. **Notifications** — submission-received (to student + Glen) and approved (to student) via the
   existing email helpers. Best-effort; never block the request.
8. **Tests** — store unit tests (in-memory sqlite), `cert_rules.evaluate` unit tests, route tests
   (monkeypatched), following `tests/test_cert_bonus.py` / `tests/test_cert_student.py`.

### Out of scope (v2+, note in code)

- Public practitioner **spotlight page** and **finder badge** UI (publish writes the data + the
  case-study proof now; the dedicated spotlight surface + badge rendering come later).
- The **interview/feature** workflow for standouts (manual for now).
- Heavy **video hosting** — intake is link-first (social/YouTube/article URLs count as
  published); file upload is limited to written docs/images (reuse the biofield photo pattern).
- Migrating the existing third-party feedback-portal submissions in automatically.

## Data model

`cert_submissions` (sqlite, in `chat_log.db`):

| column | type | notes |
|---|---|---|
| `id` | TEXT PRIMARY KEY | uuid4 |
| `email` | TEXT | student email (lowercased); links to practitioner record |
| `practitioner_id` | TEXT | resolved at submit time (nullable if not yet a practitioner) |
| `title` | TEXT | short title |
| `description` | TEXT | what it is / context |
| `url` | TEXT | link to the published work (may be empty if file-only) |
| `file_path` | TEXT | private path under DATA_DIR if a file was uploaded (nullable) |
| `formats` | TEXT | JSON list of format keys (from `cert_rules.FORMATS`) |
| `format_other` | TEXT | free-text "other" format |
| `modules` | TEXT | JSON list of module ids 1..12 the student checked |
| `module_other` | TEXT | free-text "other" topic |
| `topic_angle` | TEXT | optional (§2 angle) |
| `permission` | INTEGER | 1 if the student granted publish+reuse permission |
| `status` | TEXT | `submitted` → `approved` → `published`; plus `returned` |
| `credited_modules` | TEXT | JSON list of module ids Glen confirms on approve (defaults to `modules`) |
| `review_note` | TEXT | reviewer note (e.g. why returned) |
| `case_study_id` | TEXT | Pinecone vector id once published (nullable) |
| `created_at` / `updated_at` | TEXT | ISO-UTC |

Status flow: `submitted → approved → published`; `submitted → returned` (student can resubmit;
re-open to `submitted`). Only `approved`/`published` submissions count toward the rules; only
`credited_modules` of those count for coverage.

## Components & flow

### `dashboard/cert_rules.py` (pure, no I/O)

- `MODULES`: ordered list of the 12 `{id, label}` (Body … Prognosis/belief — verbatim from the
  framework doc §3).
- `FORMATS`: list of `{key, label, kind}` where `kind ∈ {"written","video","audio","visual"}`
  (the §1 menu, each tagged so we can test the written+video rule).
- `MIN_SUBMISSIONS = 12`.
- `evaluate(submissions: list[dict]) -> dict` — given approved/published submissions (each with
  `credited_modules` + `formats`), return:
  `{approved_count, modules_covered:set, modules_missing:list, has_written:bool, has_video:bool,
    multi_modality:bool, complete:bool, reasons:list}`. `complete` is the AND of all rules.
- This is the single place the rules live; editing it changes the program.

### `dashboard/cert_submissions.py` (sqlite store)

CRUD mirroring `cert_bonus.py`: `init_tables(cx)`, `create(cx, **fields) -> id`,
`get(cx, id)`, `list_for_email(cx, email)`, `list_by_status(cx, status=None)`,
`set_status(cx, id, status, *, credited_modules=None, review_note=None, case_study_id=None)`,
`update_fields(cx, id, **fields)`. Idempotent, `cx.commit()` after writes, `dict(row)` reads.

### Student portal

- `POST /cert/login` `{email}` — always 200 (no enumeration). Reuse `send_magic_link_email`
  with `purpose="cert_portal"` and `magic_url = {PUBLIC_BASE_URL}/cert/auth/<token>`. (Use the
  generic `auth_tokens` table via the existing `_hash_token` + insert helpers.)
- `GET /cert/auth/<token>` — validate + consume, set HttpOnly cookie `rm_cert_email`, redirect
  `/cert`. Helper `_cert_email_from_cookie()` reads it; `/cert/*` submit APIs 401 without it.
- `GET /cert` — `static/cert-portal.html`. Unauth state = email entry; authed state = the
  submission form + a list of the student's submissions with status, plus a live progress readout
  from `cert_rules.evaluate` (modules covered X/12, written ✓/✗, video ✓/✗, approved N/12).
- `POST /api/cert/submit` (authed by `rm_cert_email`) — body = title, description, url, formats[],
  format_other, modules[], module_other, topic_angle, permission(bool). Validates permission is
  granted, at least one of {url, file}, ≥1 module checked. Resolves `practitioner_id` by email
  (best-effort). Creates a `submitted` row. Fires submission-received emails (best-effort).
  Returns the new submission.
- `POST /api/cert/upload` (authed) — optional light file (image/PDF), ≤10MB, MIME-checked, stored
  to a private dir under DATA_DIR with a hashed filename (mirror `POST /api/biofield/photo`).
  Returns a file token the submit call references. Bytes never web-served.
- `GET /api/cert/mine` (authed) — the student's submissions + their `cert_rules.evaluate` rollup.

### Console review surface

- `GET /console/cert` — `static/console-cert.html` (no-cache). Lists submissions (filter by
  status), shows tags + link/file, and per-student progress. Console-key gated like other boards.
- `POST /api/cert/review/approve` (console-gated, CONSOLE_SECRET) — `{id, credited_modules[],
   note?}` → `set_status(approved, credited_modules=…)`; recompute the student's covered-module
  count from their approved+published submissions and sync `modules_completed` on the practitioner
  record; fire the approved email. (Gate 1.)
- `POST /api/cert/review/return` (console-gated) — `{id, note}` → `set_status(returned, …)`.
- `POST /api/cert/review/publish` (console-gated) — `{id}` → require `status=approved` +
  `permission=1`; embed the title+description (+url) and upsert one vector into the `case-studies`
  namespace with metadata `{title, name: practitioner_name, url, source:"cert-submission"}`; store
  the returned vector id as `case_study_id`; `set_status(published)`. (Gate 2.)
- `GET /api/cert/review/list` (console-gated) — all submissions (optionally `?status=` / `?email=`)
  plus per-student rollups, for the board to render.

### Publish to case-studies (reuse the existing proof path)

Mirror the generic upsert in app.py (~10170): build embedding for the submission text, then
`_idx.upsert(vectors=[{id, values, metadata}], namespace="case-studies")`. The chat/funnel already
reads this namespace via `_query_proof_cards` (app.py:972), so no read-side change is needed.
Vector id = `cert-<submission_id>`. Unpublish (if ever needed) = delete by that id (v2; not built).

## RBAC / auth summary

- Student-facing submit/read endpoints: gated by the `rm_cert_email` magic-link cookie.
- Console review endpoints: gated by `CONSOLE_SECRET` (X-Console-Key header or `?key=`), matching
  `POST /api/cert/student`. Owner/Ops only in practice. (No public exposure.)

## Error handling

- All student endpoints validate inputs and return JSON `{ok:false,error}` with 400 on bad input,
  401 without the cookie. Email enumeration avoided on login (always 200).
- Email + Pinecone calls are best-effort and wrapped so a failure never 500s the user action
  (log + continue); publish returns an error only if the upsert itself fails (so Glen can retry).
- File upload validates MIME + size; never serves bytes; stores under DATA_DIR.

## Testing

- `tests/test_cert_rules.py` — `evaluate` unit tests: incomplete (missing modules / no video /
  <12), complete, multi-module-single-submission coverage, written+video detection by `kind`.
- `tests/test_cert_submissions.py` — store CRUD + status transitions (in-memory sqlite).
- `tests/test_cert_portal_routes.py` — login always 200; auth sets cookie + redirects, rejects
  expired/consumed; submit 401 without cookie, validates permission + module + link/file; review
  endpoints console-gated; approve syncs modules_completed (monkeypatched); publish requires
  approved+permission and calls the (monkeypatched) upsert.
- Stub Pinecone/embeddings/email + practitioner-record helpers via monkeypatch; sqlite to tmp.
- Run: `doppler run -p remedy-match -c prd -- env DATA_DIR="$HOME/deploy-chat"
  ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_cert_*.py` (ignore the 2 known
  pre-existing failures). Pure modules (`cert_rules`, `cert_submissions`) also run under the bare
  venv.

## Files

- Create: `dashboard/cert_rules.py`, `dashboard/cert_submissions.py`,
  `static/cert-portal.html`, `static/console-cert.html`,
  `tests/test_cert_rules.py`, `tests/test_cert_submissions.py`, `tests/test_cert_portal_routes.py`.
- Modify: `app.py` — the student portal routes, the upload route, the console review routes, the
  publish-to-case-studies helper, the modules_completed sync on approve, and store-table init at
  startup (next to the other `init_tables` calls). Register `/cert` + `/console/cert` page routes.

## Feature flag

Gate the new routes behind `CERT_PORTAL_ENABLED` (env, default off), mirroring the other ladder
flags, so this ships dark and Glen flips it live in Render when ready. The student portal page and
all `/api/cert/submit|upload|mine` + review endpoints check the flag; console student-management
(`/api/cert/student`) is unaffected.

## Open / deferred

- Spotlight page + finder badge rendering (v2).
- Auto-pulling the third-party feedback portal submissions (manual cross-entry for now).
- "Certified" milestone action (when `evaluate.complete` first turns true) — v1 surfaces it in the
  console; the formal cert grant/announcement is a later step.
