# Community — coach volunteer directory (PB→illtowell Community, coaching arc, slice 1) — Design

**Date:** 2026-07-05
**Status:** Approved in brainstorm with Glen 2026-07-05 (students volunteer + intro video; members browse a coach directory; request+accept and the 1:1 thread are later slices).
**Repo:** deploy-chat

**Relates to / reuses:**
- **Practitioners (Postgres/Supabase):** `dashboard/practitioner_admin.py` / `practitioner_portal.py` — coaches are `practitioners.portal_role='coach'`; `modules_completed`, `credentials`, `name`, `email`.
- **Cert eligibility:** `dashboard/cert_rules.py:evaluate(submissions)` (→ `complete` flag), `dashboard/cert_submissions.py`.
- **Member coaching eligibility:** `dashboard/coaching.py:active_window(cx, email)` (a member's active coaching window).
- **Video hosting:** Rumble unlisted (see [[reference_video_hosting_rumble]]) — the intro video is a Rumble URL on the coach card.
- `_evox_ident` (member portal-token auth), `CONSOLE_SECRET`, `LOG_DB`, `_db_lock`.
- [[project_pb_to_illtowell_evox]], [[project_ash_five_fold_dimensions]] (ASH certification).

## Context and boundary

Glen wants certification students to gain coaching experience by coaching paid members: **students volunteer**, record an **intro video**, members **browse and pick** a coach, and **students pick** whom to accept; then they talk in a 1:1 thread. This is the coaching arc, distinct from the C2a peer-matching arc (on hold). It decomposes into:
- **Slice 1 (THIS SLICE):** coach volunteer signup + intro video + the member-facing coach directory.
- **Slice 2 (deferred):** member requests a coach → student accepts up to capacity (the pairing).
- **Slice 3 (deferred):** the 1:1 coaching thread + report/block + moderation (shared channel with C2a peer matching).

**Data-model reality:** the coach roster + cert status live in **Postgres** (`practitioners`); Community data lives in **sqlite** (`chat_log.db`). This slice keeps the member-facing directory on sqlite by storing a **volunteer profile** per coach (denormalizing name + focus + video at signup), so the directory hot path does not query Postgres.

## Scope

**A coach volunteers (profile + intro video) → members with an active coaching window browse the directory.** One volunteer-profile store + a signup endpoint (with cert-eligibility check) + the member-facing directory route + a directory card in the portal.

**Deferred:** request/accept pairing (slice 2), the 1:1 thread + report/block/moderation (slice 3), coach self-service signup via the practitioner portal (see the signup decision below), capacity enforcement at pairing time.

## Components

### 1. Volunteer profile store (`dashboard/coach_directory.py`)

- `coach_volunteers(email TEXT PRIMARY KEY, name TEXT, focus TEXT, intro_video_url TEXT, capacity INTEGER DEFAULT 3, active INTEGER DEFAULT 1, cert_ok INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT)` — one row per volunteering coach (keyed by their practitioner email). `focus` = a short blurb; `intro_video_url` = a Rumble unlisted link; `capacity` = how many members they'll take (used at pairing time in slice 2, stored now); `active` = currently listed; `cert_ok` = eligibility snapshot at signup.
- Functions (pure sqlite, emails lowercased, no app imports): `init_coach_tables(cx)`; `upsert_volunteer(cx, *, email, name, focus, intro_video_url, capacity, cert_ok) -> None`; `set_active(cx, email, active)`; `get_volunteer(cx, email) -> dict|None`; `list_active(cx) -> [dict]` — active AND cert_ok volunteers, newest first, returning the **member-safe fields only**: `{name, focus, intro_video_url}` (NO email, NO capacity, NO raw row). `credentials` is added by the route from the practitioner record if available. A coach volunteering to be listed consents to being shown (see privacy note).

### 2. Signup + eligibility (`app.py`)

- **Eligibility check** `_coach_cert_ok(email) -> bool`: the practitioner is `portal_role='coach'` AND `cert_rules.evaluate(their submissions)['complete']` (best-effort; on any lookup error → False, fail-closed — an unverified student is NOT listed). Reads Postgres/`cert_submissions`.
- **Signup endpoint** `POST /api/console/coach-volunteers {email, focus, intro_video_url, capacity}` (**CONSOLE_SECRET-gated for slice 1** — see decision): looks up the practitioner name, runs `_coach_cert_ok`, `upsert_volunteer(cert_ok=…)`. A coach who is not cert-eligible is stored `cert_ok=0` and NOT listed. Returns `{ok, cert_ok, listed}`.
- **Signup auth decision (FLAGGED for review):** slice 1 makes volunteer signup **console-gated** rather than practitioner-portal self-service. Rationale: it decouples this slice from wiring the practitioner-portal session auth, and it doubles as a human gate on who coaches paying clients. **Coach self-service signup via the practitioner portal is the immediate next step (slice 1b).** If you'd rather have self-service in this slice, say so and I'll spec the practitioner-portal-authed variant instead.

### 3. Member-facing directory (`app.py`)

- `GET /api/community/coaches?token=…` (member portal-token via `_evox_ident`; bad token → 404). Gate: the member must have an **active coaching window** (`coaching.active_window`) — else `{eligible: false, coaches: []}` (the directory is for members entitled to coaching). Eligible → `{eligible: true, coaches: [{name, focus, intro_video_url}]}` from `list_active`. No coach email is exposed to the member in this slice (pairing/contact is slice 2+). The surface is the portal card (§4), fed by this API — no separate page.

### 4. Member surface (`static/client-portal.html`)

- A "Meet your coaches" card, shown only to members with an active coaching window: each active volunteer as a card with name, focus, credentials, and the embedded Rumble intro video. A quiet line: "Choosing your coach is coming soon." (Request/accept is slice 2.) Names/focus via `textContent`; the video via a Rumble `<iframe>` from `intro_video_url`. Copy: no em dashes, no ALL CAPS.

## Config

No new required env. Reuses Postgres practitioner access, `cert_rules`, `coaching.active_window`, Rumble, `CONSOLE_SECRET`.

## Privacy

- The directory intentionally exposes a volunteering coach's name, focus, credentials, and intro video — a coach who volunteers to be listed consents to being seen by members. This is the one place a person is shown to others, and it is opt-in by the coach.
- A coach's **email is never exposed** to members in this slice (no contact until pairing in slice 2+).
- Only members with an active coaching window see the directory; non-eligible members get an empty/ineligible response.
- Cert eligibility is fail-closed: a student who cannot be verified as cert-complete is never listed.

## Testing

- Pure/sqlite (`dashboard/coach_directory.py`): `upsert_volunteer` insert+update (idempotent on email); `list_active` returns only `active=1`, newest first, and does NOT include `email` in the member-safe fields; `set_active` toggles listing; `get_volunteer` round-trip.
- Route/api: signup is CONSOLE_SECRET-gated (401 without); a cert-ineligible email is stored `cert_ok=0` and NOT listed (mock `_coach_cert_ok`); a cert-eligible email is listed. `GET /api/community/coaches` — a member with an active coaching window sees active coaches (name/focus/video, NO email); a member without a window gets `{eligible:false, coaches:[]}`; bad token → 404. `_coach_cert_ok` fails closed (returns False) when the practitioner/cert lookup errors.
- Regression: Community A/B/C1/C3 untouched; the directory reads coaching/practitioner data but writes only its own table.
- Go-live: enroll one cert-eligible coach with a Rumble intro video; a member with an active coaching window sees the coach card + video; a member without a window does not; an ineligible coach is not listed.

## Deferred (coaching arc, later slices)

- **Slice 1b:** coach self-service volunteer signup via the practitioner portal (replacing/adding to the console-gated signup).
- **Slice 2:** member requests a coach → student accepts up to `capacity` (the pairing); reveal contact/enable the thread.
- **Slice 3:** the 1:1 coaching thread + report + block + a moderation surface (shared channel with the C2a peer-matching arc).
- Coach-side dashboard (see/manage requests and current coachees), interest-matching of coaches to members, capacity enforcement.
