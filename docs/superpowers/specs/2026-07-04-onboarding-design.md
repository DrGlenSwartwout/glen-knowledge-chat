# New-Member Onboarding — member phone call (PB→illtowell appointment loop, slice 5) — Design

**Date:** 2026-07-04
**Status:** Approved in brainstorm with Glen 2026-07-04 (pull-first confirmed).
**Repo:** deploy-chat
**Session:** PB→illtowell · EVOX (continued)

**Relates to / reuses:**
- `dashboard/evox.py` booking engine (`create_booking`, `available_slots`, `rae_busy_intervals`, `booked_starts`, `SlotTaken`, `build_ics`) — same engine EVOX/Consult/Triage use.
- `_is_paid_member(email)` (app.py:5120) — the member gate.
- `_evox_ident` (portal-token → identity), `client_portal.ensure_token`, `send_evox_email`, `PUBLIC_BASE_URL`, `EVOX_HOURS` (Rae's office hours), `_hst_now`, the `glen-evox-reminders` cron + `/api/evox/run-reminders` copy dispatch, `dashboard/portal_view.py:get_portal_view`, `static/client-portal.html`.

## Summary

Give a new **paid member** a free 15-minute **phone** welcome call with Rae: orient them to the portal and their membership, answer early questions. It is the lightest slice of the appointment loop — it adds no Zoom, no Stripe, no new env. It reuses the EVOX booking engine end to end and differs from the other slices only in its **gate** (paid-membership, self-serve) and its **once-per-member** rule.

Shape: **Triage, but self-serve for members instead of invite-only.** A member opens their portal, sees a "Book your welcome call" card, picks a 15-minute slot on Rae's availability, and gets a phone-call confirmation + calendar invite. Rae calls them at the booked time.

## Scope

**Member sees card → picks slot → books → phone confirmation + ICS → Rae calls.** No payment (member benefit, `amount=0`), no Zoom. Once a member has a booked onboarding call, the card shows "You're booked for &lt;time&gt;" and offers no new slots.

**Deferred / out of scope:**
- **Push discovery** — auto-emailing a "book your welcome call" invite when someone becomes a paid member. Pull-first (portal card only) ships now; push is an easy fast-follow once we see self-serve uptake. (Glen chose pull-first.)
- Rescheduling/cancel from the portal, post-call notes, and any onboarding checklist/content beyond the call itself.

## Components

### 1. Onboarding config + existing-booking helper

- New module `dashboard/onboarding.py` (small, mirrors `consult.py`'s shape; unit-testable in isolation):
  - `ONBOARDING = {"session_type": "onboarding", "practitioner": "rae", "medium": "phone", "duration_min": 15}`.
  - `existing_onboarding(cx, email) -> dict | None` — returns the member's currently **booked** onboarding row (`session_type='onboarding' AND status='booked'`, email lowercased) or `None`. Used by both the state card and the book short-circuit (the once-per-member rule).
- The module does **not** know about membership; the route owns the `_is_paid_member` gate (keeps the module free of app-layer imports, like `consult.py`).

### 2. Calendar label fix (folds in the triage cosmetic fast-follow)

- `create_booking` (`dashboard/evox.py:185`) currently labels calendar rows `"Biofield Consult"` for consult and `"EVOX"` for everything else — so triage and onboarding both mislabel as "EVOX". Replace the single ternary with a session-type → label map:
  `{"biofield-consult": "Biofield Consult", "triage": "Discovery Call", "onboarding": "Welcome Call"}`, default `"EVOX"`. Onboarding rows show "Welcome Call", triage rows show "Discovery Call", EVOX unchanged. No signature change; no caller change.

### 3. Booking routes (portal-token gated, member-gated)

All three authed via `_evox_ident(cx, token)` (portal token), then `_is_paid_member(ident.email)`:

- `GET /api/onboarding/state?token=…` → `{eligible: bool, booked: {start_ts} | null}`. `eligible` = `_is_paid_member`. If an `existing_onboarding` row exists, `booked` carries its `start_ts` (card shows the confirmed time, no picker). Non-member → `{eligible: false}` (200; the card simply doesn't render for them).
- `GET /api/onboarding/availability?token=…` → `{slots: [...]}` — Rae's `EVOX_HOURS`, `rae_busy_intervals`/`booked_starts` for `practitioner='rae'`, 15-min grid, forward-looking from `_hst_now()`. Member-gated (403 if not a member); returns `{slots: []}` if the member already has a booked onboarding (once-per-member).
- `POST /api/onboarding/book {start_ts}` (token in query) → member-gated (403 non-member); **short-circuit 409 `already_booked`** if `existing_onboarding` is non-null; validate `start_ts` is in the live availability set (409 `slot_unavailable`); `create_booking(cx, email, start_ts, duration_min=15, practitioner='rae', session_type='onboarding', medium='phone')` (no `tag_fn` — matches consult); on `SlotTaken` → 409 `slot_taken`. Zoom/Stripe: none. Returns `{ok: true, start_ts}`.

Lock discipline mirrors consult/triage: DB work under `with _db_lock, sqlite3.connect(LOG_DB)`; the confirmation email send runs **after** the lock is released.

### 4. Delivery + portal surface

- `_onboarding_send_confirmations(email, booking)` (best-effort, after lock): phone-call confirmation to the member (date/time HST, "Rae will call you at the number on file", the portal link) + an ICS invite (`build_ics`, `location="Phone"`). Best-effort — a send failure never unwinds the durable booking.
- Reminder copy: `/api/evox/run-reminders` already branches copy on `session_type`; add an `onboarding` branch (phone "Rae will call you" welcome-call copy, distinct from the EVOX ZYTO phone copy and the consult Zoom copy). The existing `glen-evox-reminders` cron covers onboarding automatically (they are `evox_bookings` rows).
- `_onboarding_block(cx, email)` in `dashboard/portal_view.py`, wired into `get_portal_view`'s returned dict as `"onboarding"`: `{eligible, booked_start}` for the template. Card + 15-min slot picker in `static/client-portal.html` (reuses the consult card's fetch/pick/submit pattern; success keyed on `d.ok`). Card renders only when `eligible`; shows the picker when not yet booked, the confirmed time when booked.

## Config

No new env. Reuses `EVOX_HOURS` (Rae's hours + phone), `EVOX_RAE_PHONE`/`EVOX_RAE_EMAIL`, `PUBLIC_BASE_URL`, the existing reminder cron.

## Copy guidance

Client-facing copy: no em dashes, no ALL CAPS. Warm, welcoming — this is a member's first personal touch. Rae calls them; the copy says so plainly.

## Testing

- Pure/sqlite (`dashboard/onboarding.py`): `existing_onboarding` returns None when no booking, the row when a booked onboarding exists, ignores other session types and cancelled rows, lowercases email.
- Calendar label: `create_booking` writes "Welcome Call" for `session_type='onboarding'`, "Discovery Call" for `'triage'`, "Biofield Consult" for consult, "EVOX" for evox (regression).
- Route/api: `state` (member with/without booking; non-member → `eligible:false`); `availability` (member gets slots; non-member 403; already-booked member → empty slots); `book` (member free-books → `ok`, confirmation sent, `existing_onboarding` now set; second book → 409 `already_booked`; non-member → 403; unavailable slot → 409). Email mocked.
- Regression: EVOX/consult/triage/masterclass untouched except the additive label map; `create_booking` signature unchanged.
- Go-live: as a paid member, open the portal, see the card, book a slot, confirm the phone confirmation + calendar invite + the call on Rae's console calendar lane labeled "Welcome Call"; confirm a non-member sees no card.
