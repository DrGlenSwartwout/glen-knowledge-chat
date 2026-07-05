# MasterClass — one-to-many event registration (PB→illtowell appointment loop, slice 4) — Design

**Date:** 2026-07-04
**Status:** Approved in brainstorm with Glen 2026-07-04.
**Repo:** deploy-chat
**Session:** PB→illtowell · EVOX (continued)

**Relates to / reuses:**
- `dashboard/zoom.py:create_meeting` (Server-to-Server OAuth Zoom **Meeting** creation; needs a `waiting_room` param added), `dashboard/zoom.py:get_token`, Zoom creds in env.
- `_is_paid_member(email)` (member-tiered pricing), the Stripe checkout + `/webhook/stripe` `checkout.session.completed` → `_fulfill_*` dispatch (paid registration), the email/ICS rail (`send_evox_email`, `build_ics`), `_portal_console_ok()` (console auth), `PUBLIC_BASE_URL`, `STATIC`, `calendar_events` `glen` lane.

## Summary

Let Glen run a **MasterClass**: a scheduled one-to-many event (60 min, video Meeting) that many people register for. He creates it in the console (topic, time, two prices); the system creates the Zoom Meeting and gives him a public event page. People register via that page; the price they pay is **member-tiered** (`_is_paid_member` → member price, often free, vs non-member price), paid via Stripe when non-zero. Registrants get the Zoom join link (gated to them). Glen announces the event-page link through GoHighLevel.

This is a **different shape** from the 1:1 appointment slices (EVOX/Consult/Triage): it does NOT reuse the slot-booking engine; it has its own event + registration storage.

## Scope

**Create → event page → register (member-tiered, free/paid) → deliver join link.** One new module + a console create form + a public event page + a Stripe fulfillment handler.

**Deferred / out of scope:** system-sent list announcements (Glen uses GHL), capacity/waitlist, editing a class after creation, and posting the recording afterward.

## Components

### 1. Event store

- `masterclass_events(id INTEGER PK, topic TEXT, description TEXT, start_ts TEXT, duration_min INTEGER DEFAULT 60, price_cents INTEGER DEFAULT 0, member_price_cents INTEGER DEFAULT 0, zoom_join_url TEXT, zoom_meeting_id TEXT, created_at TEXT)`. (`start_ts` naive ISO HST.)
- `masterclass_registrations(id INTEGER PK, event_id INTEGER, email TEXT, name TEXT, is_member INTEGER, amount_cents INTEGER, paid INTEGER DEFAULT 0, created_at TEXT, UNIQUE(event_id, email))`.
- Module `dashboard/masterclass.py`: `init_masterclass_tables(cx)`; `create_event(cx, *, topic, description, start_ts, duration_min, price_cents, member_price_cents) -> event_id`; `get_event(cx, event_id) -> dict|None`; `set_zoom(cx, event_id, join_url, meeting_id)`; `price_for(event, is_member) -> int` (member_price_cents if member else price_cents); `register(cx, event_id, email, name, is_member, amount_cents, *, paid) -> None` (upsert on `(event_id, email)`); `mark_paid(cx, event_id, email)`; `is_registered(cx, event_id, email) -> bool`.

### 2. Console create

- Console-gated `POST /api/console/masterclass {topic, description, start_ts, duration_min, price_cents, member_price_cents}` (`_portal_console_ok`): `create_event`; then **best-effort** create the Zoom Meeting (`get_token`+`create_meeting(..., waiting_room=False)`), `set_zoom` with the join URL/id; a Zoom failure does NOT block event creation (returns `zoom_ok:false` so Glen can paste a link). Also `POST /api/console/masterclass/<id>/zoom-url {url}` to set the link manually (fallback). Returns `{ok, event_id, event_url: PUBLIC_BASE_URL+/masterclass/<id>, zoom_ok}`. Surfaced as a "Create MasterClass" form on a console page.
- The Zoom `create_meeting` gets a new `waiting_room: bool = True` param (default preserves EVOX/consult behavior; MasterClass passes `False` — registration is the gate, not the waiting room).

### 3. Event page + registration APIs (public)

- `GET /masterclass/<id>` serves `static/masterclass.html`.
- `GET /api/masterclass/<id>` → `{topic, description, start_ts, duration_min, price_cents, member_price_cents}` (public; no join link).
- `POST /api/masterclass/<id>/register {email, name}` → compute `is_member = _is_paid_member(email)`, `amount = price_for(event, is_member)`:
  - `amount == 0` → `register(..., paid=True)`, send confirmation with the join link, return `{ok, registered:true, join_url}` (the just-registered person gets the link immediately).
  - `amount > 0` → create a Stripe checkout session (metadata `{kind:'masterclass', event_id, email, name}`), `register(..., paid=False)` (pending), return `{ok, checkout_url}`.
- **Delivery of the Zoom link is by email + the immediate register response** — there is no standalone email-keyed join endpoint (which would be a weak gate). Free/member registrants get `join_url` in the register response *and* the confirmation email; paid registrants get it in the confirmation email after Stripe success. (Return-visit link display on the page is deferred.)
- Stripe fulfillment: add `_fulfill_masterclass(session_id)` to the `/webhook/stripe` `checkout.session.completed` dispatch — reads `metadata.event_id/email`, `mark_paid`, sends the confirmation with the join link.

### 4. Delivery + calendar

- Confirmation email (best-effort) to the registrant: topic, date/time HST, the **Zoom join link**, and an ICS invite. Practitioner (Glen) sees registrations; a MasterClass shows on Glen's console calendar lane as one event (a synthetic `glen`-lane `calendar_events` row at create time).
- **Free** and **member-free** register immediately; **non-member paid** registers on Stripe success.

## Config

- No new required env. Reuses Zoom creds, Stripe keys, `PUBLIC_BASE_URL`, `GLEN_CONSULT_EMAIL`. **Prerequisite: the Zoom S2S app must be enabled for auto meeting-creation** (manual-URL fallback otherwise).

## Settled decisions

- **System creates the Zoom Meeting** (not webinar), waiting room off, cloud recording on. Manual-URL fallback if the API fails.
- **Member-tiered pricing per event:** member price (default $0, free with membership) vs non-member price; decided at registration by `_is_paid_member`. Free class = both $0. Paid non-member → Stripe.
- **Glen announces via GHL** with the event-page link; the system does not send list broadcasts.
- One-to-many event registration; own storage (not the slot-booking engine). 60-min default.

## Copy guidance

Client-facing copy: no em dashes, no ALL CAPS. Warm, inviting.

## Testing

- Pure/sqlite: `create_event`/`get_event`/`set_zoom`/`price_for`(member vs non)/`register`(upsert)/`mark_paid`/`is_registered`.
- Route/api: console create (auth-gated; creates event; Zoom best-effort with a mocked `create_meeting`; `zoom_ok` reflects success/failure); manual zoom-url set; public `GET /api/masterclass/<id>`; register — free → registered immediately + confirmation; member → member price (0 → free); non-member paid → returns `checkout_url` + pending registration; `_fulfill_masterclass` marks paid + sends the join link. Free/member register response carries `join_url`. Stripe + Zoom mocked.
- Regression: EVOX/consult/triage untouched; `create_meeting`'s new `waiting_room` param defaults to the old behavior.
- Go-live: enable the Zoom S2S app; create one real MasterClass, register as a member (free) and as a non-member (Stripe), confirm the join link + calendar invite + the class on Glen's console calendar.
