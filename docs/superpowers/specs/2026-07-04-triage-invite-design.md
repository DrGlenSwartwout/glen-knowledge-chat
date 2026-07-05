# Triage / Discovery booking (PB→illtowell appointment loop, slice 3) — Design

**Date:** 2026-07-04
**Status:** Approved in brainstorm with Glen 2026-07-04.
**Repo:** deploy-chat
**Session:** PB→illtowell · EVOX (continued)

**Relates to / reuses:**
- EVOX (slice 1) + Biofield Consult (slice 2, incl. static-room #594): the booking engine `dashboard/evox.py` (`create_booking(session_type, practitioner, medium, ...)`, `available_slots`, `booked_starts`, `rae_busy_intervals`, `SlotTaken`), the confirmation-email/ICS rail (`send_evox_email`, `build_ics`), `calendar_events` `glen`/`rae` lanes, and the consult join-gate primitives `dashboard.consult.within_join_window` + `GLEN_PMI_URL`.
- `EVOX_RAE_PHONE` (Rae's number, phone medium), `GLEN_CONSULT_HOURS` / `EVOX_HOURS` (per-practitioner hours), `_portal_console_ok()` (console auth).

## Summary

Let Glen or Rae **invite a prospect** (mostly new, no account) to a free **15-minute Triage / Discovery call**, by email, at their discretion. The prospect books a slot from the assigned practitioner's real availability via a tokenized booking page (no login). Rae's triage is by phone (the prospect calls Rae); Glen's is by Zoom (a time-gated join button on the booking page, reusing the consult pattern). This is the front-of-funnel, invite-only member of the appointment loop.

## Scope

**Invite → book → connect.** One new invite store + a tokenized booking page + booking on the existing engine (`session_type='triage'`). Practitioner is chosen **at invite time** (Glen or Rae). Availability **reuses each practitioner's existing hours** on a 15-minute grid.

**Deferred / out of scope:** MasterClass (one-to-many event shape), the intake form, prospect-portal creation, any public (non-invite) triage page.

## Components

### 1. Invite store + console invite

- New table `triage_invites(token_hash TEXT PRIMARY KEY, email TEXT, name TEXT, practitioner TEXT, status TEXT DEFAULT 'invited', created_at TEXT, expires_at TEXT, booked_start TEXT)`. Raw token minted with `secrets.token_urlsafe(24)`, only its `sha256` stored (same trust model as portal tokens).
- Module `dashboard/triage.py`: `init_triage_tables(cx)`; `create_invite(cx, email, name, practitioner, *, days=7) -> raw_token`; `resolve_invite(cx, token) -> dict|None` (returns the invite if the hash matches, status != 'cancelled', and not expired); `mark_booked(cx, token, start_ts)`.
- Console-gated `POST /api/console/triage-invite {email, name, practitioner}` (`_portal_console_ok`): validates `practitioner in ('glen','rae')`, `create_invite`, emails the prospect the booking link `{PUBLIC_BASE_URL}/triage/<raw_token>` (via `send_evox_email`, no ICS), returns `{ok, url}`. Surfaced as a small "Send triage invite" form on an existing console page (e.g. `console-biofield-portal.html` or the console home).

### 2. Booking page + invite-token-authed APIs

- `GET /triage/<token>` serves `static/triage.html`. All `/api/triage/*` authenticate on the **invite token** (via `resolve_invite`), NOT a portal token.
  - `GET /api/triage/state?token=` → `{name, practitioner, medium, booked, booked_start}` (404 `invalid` if the invite doesn't resolve / expired).
  - `GET /api/triage/availability?token=&range=week` → `{slots}` for the invite's practitioner, 15-min grid from that practitioner's hours (`GLEN_CONSULT_HOURS` if glen else `EVOX_HOURS`) minus that practitioner's `rae_busy_intervals`/`booked_starts`. 409 `already_booked` if the invite is already booked.
  - `POST /api/triage/book?token= {start_ts}` → server-side re-validate the slot; `create_booking(session_type='triage', practitioner=<invite>, duration_min=15, medium=('phone' if rae else 'video'))`; `mark_booked`; send confirmations; return `{ok, start_ts}`. `SlotTaken`→409.
  - `GET /api/triage/join?token=` (Glen/video only) → returns `{ok, join_url: GLEN_PMI_URL}` iff the invite is booked and `within_join_window(booked_start, _hst_now())`, else 403 `not_in_window` / 404. (Rae/phone invites: this route returns 400 `phone_call`.)
- `static/triage.html`: a small self-contained page (like `evox.html`). Shows "Book your 15-minute call with Dr. Glen / Rae", a slot picker, then a confirmed state: for **Rae** "call Rae at [number] at [time]"; for **Glen** a time-gated "Join your call" button calling `/api/triage/join`.

### 3. Booking, confirmation, connect

- Booking writes an `evox_bookings` row (`session_type='triage'`, the practitioner, `medium`) + the synthetic `glen`/`rae`-lane calendar row (via the existing `create_booking`). Cross-session conflicts (a triage overlapping an EVOX/consult for the same practitioner) are prevented by the same busy-and-booked availability check + the partial unique index `(practitioner, start_ts) WHERE status='booked'`.
- Confirmations (best-effort, no raw Zoom link):
  - **prospect:** Rae → "At your appointment time, call Rae at `EVOX_RAE_PHONE`." Glen → "At your appointment time, open this page and click Join your call: `{PUBLIC_BASE_URL}/triage/<token>`." Plus an ICS invite either way.
  - **practitioner:** a notice to `EVOX_RAE_EMAIL` (rae) or `GLEN_CONSULT_EMAIL` (glen): "New triage booked: [name] <email> on [time]."
- **Free** (no payment). Invite is **single-use** — booking sets `status='booked'`; a second attempt returns `already_booked`.

## Data model

- New `triage_invites` (above). No change to `evox_bookings` (reuses `session_type`/`medium`), `calendar_events`.

## Config

- No new required env. Reuses `GLEN_CONSULT_HOURS`, `EVOX_HOURS`, `EVOX_RAE_PHONE`, `GLEN_PMI_URL`, `GLEN_CONSULT_EMAIL`, `EVOX_RAE_EMAIL`, `PUBLIC_BASE_URL`. Invite expiry default **7 days**.

## Settled decisions

- Per-person **email invites**, prospect books via tokenized page (no account). Practitioner **assigned at invite time**. Free. Availability **reuses each practitioner's hours**, 15-min grid. Invite **7-day** expiry, single-use.
- Medium is **per-practitioner**: **Rae = phone (prospect calls Rae)**; **Glen = Zoom (time-gated Join button on the booking page, reusing `within_join_window` + `GLEN_PMI_URL`)**. Raw Zoom link never emailed.

## Copy guidance

Client-facing copy: no em dashes, no ALL CAPS. Warm, welcoming (front-of-funnel first impression).

## Testing

- Pure: `create_invite`/`resolve_invite` (hash match, expiry, cancelled), `mark_booked`; availability for each practitioner's hours at 15-min; the join gate for a glen triage.
- Route/api: console invite creates + emails; `/api/triage/*` gated on a valid invite token (invalid/expired → 404); availability 15-min for the right practitioner; book creates a `triage` `evox_bookings` row on the right lane, marks the invite booked, `already_booked` on re-book, `SlotTaken`→409; confirmation has no raw Zoom link (glen) / has the call-Rae number (rae); `/api/triage/join` window-gated for glen, `phone_call` for rae.
- Regression: EVOX + consult suites stay green (booking engine unchanged; only new session_type value + new module/routes).
- Go-live: send one real triage invite to each of Glen/Rae, book end to end, confirm the phone (rae) / gated Join (glen) + calendar lane.
