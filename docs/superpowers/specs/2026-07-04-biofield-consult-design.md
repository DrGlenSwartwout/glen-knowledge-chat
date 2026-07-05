# Biofield Consult booking (PB→illtowell appointment loop, slice 2) — Design

**Date:** 2026-07-04
**Status:** Approved in brainstorm with Glen 2026-07-04.
**Repo:** deploy-chat
**Session:** PB→illtowell · EVOX (continued)

**Relates to / reuses:**
- **EVOX booking (slice 1, shipped PR #582/#583):** `dashboard/evox.py` (the booking core: availability, `create_booking`, ICS, credits), the `evox_bookings` table (partial-unique `(practitioner, start_ts) WHERE status='booked'`), the confirmation-email + ICS helpers in `app.py`, and the session-type catalog seam. This slice **generalizes that engine to a second session type + practitioner.**
- Client portal: `/portal/<token>`, `dashboard/portal_identity.py:resolve_identity`, `portal_biofield_reports` (the E4L-AI report store — NOT the gate here; see below).
- Membership / payment: `_is_paid_member(email)`, the `orders` table (in-house order for the paid test).
- Calendar: `calendar_events` (`glen` lane; Glen's Google calendars are unmapped so the sync heuristic files them as `owner='glen'`).

## Summary

Let a client who has paid for and received their **Causal Biofield Analysis** book the 30-minute **Biofield Consult** with Dr. Glen (to review their proposed ASH Program), over Zoom, from inside their existing portal. This is stage 7 of the 9-stage Biofield Analysis journey; the booking is gated by a manual "consult ready" unlock that Glen/Rae flip when the Causal report + proposed-program invoice are posted.

This is the direct EVOX analog (reuses the shipped booking engine) plus three new pieces: it runs on **Glen's** calendar, it is **entitlement-gated**, and it is the **first session type that needs video (Zoom)**.

## Scope

**A+ = the consult booking + a light portal journey checklist.** Built by generalizing `dashboard/evox.py` (add `session_type` + keep the existing `practitioner` arg) and adding a Zoom integration.

**Deferred to their own slices (explicitly out of scope):**
- **C — Intake form** (migrating data collection off Practice Better): the heaviest new piece, its own sub-project.
- **D — Post-consult video + notes** posting (stage 9): extends the portal publish flow. (This slice stores the Zoom meeting id so D can pull the cloud recording later.)
- Auto-detecting the Causal report / proposed-program invoice as system artifacts (the gate is a manual flip here).
- Hard-enforcing the intake / E4L-fresh-scan prerequisites (shown as roadmap status, not blocking gates).

## The 9-stage journey (context; only stage 7 is *built* here)

1. Intake form  ·  2. E4L account + fresh scan (<10 days)  ·  3. active/paid member  ·  4. pay for the test ($300)  ·  5. test completed (manual, 1–2 days)  ·  6. Causal report + proposed-program invoice posted  ·  **7. schedule consult (BUILT)**  ·  8. complete consult  ·  9. post private consult video + notes (slice D).

The portal shows all nine as a checklist. Stage status:
- **Auto-detected on prod:** #3 active member (`_is_paid_member`), #4 test paid (an `orders` row for the paid-test SKU by that email).
- **Reflects the manual flip:** #5 test complete, #6 report posted → these are set when Glen/Rae mark the client "consult ready."
- **Roadmap only for now:** #1 intake (waits on slice C), #2 E4L fresh scan (scan-freshness detection deferred; scan data reachability from prod is unresolved — do not over-promise it).

**The Schedule button activates iff the manual "consult ready" flag is set.** The other stages inform the client where they are but do not gate the button.

## Components

### 1. Booking engine — generalize `dashboard/evox.py`

- Add a `session_type` parameter to `create_booking(cx, email, start_ts, *, session_type="evox", practitioner="rae", duration_min=60, ...)` and to the availability/query helpers as needed. Add a lazy `ALTER TABLE evox_bookings ADD COLUMN session_type TEXT DEFAULT 'evox'` (guarded). Existing EVOX rows default to `evox` (backward compatible).
- The synthetic `calendar_events` row uses `owner=practitioner` (here `'glen'`) and `google_event_id=f"{session_type}-{booking_id}"` so consult bookings show on Glen's console lane and never collide with EVOX ids.
- `booked_starts` / availability filter by `practitioner` (already do). The partial-unique index stays `(practitioner, start_ts)` — correct: Glen can't be double-booked across session types at the same start.
- **Naming note:** the module stays `dashboard/evox.py` for now (it is the booking core); a rename to `dashboard/booking.py` is a future cleanup, out of scope.

### 2. Session-type catalog entry

```
biofield-consult: {
  practitioner: "glen",
  duration_min: 30,
  medium: "video",            # Zoom (new)
  gate: "consult_ready",      # manual unlock
  payment: "included",        # in the already-paid $300
  hours_env: "GLEN_CONSULT_HOURS",
}
```

### 3. The gate — manual "consult ready" unlock

- New table `consult_eligibility(email TEXT, session_type TEXT, ready INTEGER DEFAULT 0, marked_at TEXT, PRIMARY KEY(email, session_type))`.
- Console-gated endpoint `POST /api/console/consult-ready {email, ready}` (header `X-Console-Key`) sets/clears the flag. Surfaced as a **"Mark consult ready"** button on the client in the console (Glen/Rae flip it when they post the Causal report + program invoice).
- Helper `consult_is_ready(cx, email, session_type) -> bool`.

### 4. Zoom integration — new `dashboard/zoom.py`

- Server-to-server OAuth (creds already in Doppler: `ZOOM_ACCOUNT_ID`, `ZOOM_CLIENT_ID`, `ZOOM_CLIENT_SECRET`). `get_token()` → `POST https://zoom.us/oauth/token?grant_type=account_credentials&account_id=...` with HTTP Basic (`client_id:client_secret`); cache the token in-process (~55 min).
- `create_meeting(*, topic, start_iso, duration_min, timezone="Pacific/Honolulu", host=GLEN_ZOOM_USER) -> {join_url, meeting_id, start_url}` → `POST https://api.zoom.us/v2/users/{host}/meetings` with `type=2` (scheduled), waiting room on. `GLEN_ZOOM_USER` env = Glen's Zoom user id/email (default `"me"`). **Verify the S2S app has `meeting:write` scope at build time**; if create fails, the booking still succeeds but the confirmation notes "Zoom link to follow" and logs the error (Zoom failure must not block the booking).
- Store `zoom_join_url` + `zoom_meeting_id` on the booking (new nullable columns via lazy ALTER) — the meeting id lets slice D fetch the cloud recording.

### 5. Portal surface + routes

- A **"Biofield Consult"** section in `static/client-portal.html` showing the 9-stage checklist + a "Schedule your consult" button (activates when ready), a slot picker, and a confirmed state (with the Zoom link). Reuses the EVOX page's view pattern; token auth via the portal token already present.
- Token-authed routes (mirror EVOX, gated additionally on `consult_is_ready`):
  - `GET /api/consult/state?token=` → `{stages:{member,paid,ready,...}, ready, booked}`.
  - `GET /api/consult/availability?token=&range=week` → `{slots}` (403 `not_ready` if the flag is unset).
  - `POST /api/consult/book?token= {start_ts}` → re-validate the slot server-side, `create_booking(session_type='biofield-consult', practitioner='glen', duration_min=30)`, create the Zoom meeting, store the link, send confirmations. On `SlotTaken` → 409.

### 6. Confirmation + money

- Generalize `_evox_send_confirmations` so `medium='video'` puts the **Zoom join link** (and ICS with the link as location/URL) in the client + Glen emails, instead of the "call Rae" phone text. Best-effort send, outside the DB lock (as EVOX).
- **No payment at booking** (included in the already-paid $300).

## Data model changes

- `evox_bookings`: add `session_type TEXT DEFAULT 'evox'`, `zoom_join_url TEXT`, `zoom_meeting_id TEXT` (all lazy ALTER, nullable/defaulted; backward compatible).
- New `consult_eligibility(email, session_type, ready, marked_at)`.
- No change to `calendar_events` (synthetic row as EVOX, `owner='glen'`).

## Config (env, set in Render at go-live)

- `GLEN_CONSULT_HOURS = "1-7:09:00-17:00"` (all days, 9am–5pm HST, 30-min grid — Glen's answer).
- `GLEN_ZOOM_USER` (Glen's Zoom user id/email; default `"me"`).
- Zoom creds already present.

## Settled decisions

- Video via **Zoom API, unique meeting per booking** (creds exist); Zoom failure never blocks the booking.
- Gate = **manual "consult ready" flip** (console), not auto-detected from the Causal report (which is distinct from the E4L-AI report and hand-produced).
- Objective stages (member, test paid) auto-shown; clinical stages reflect the flip; intake + E4L-scan are roadmap-only for now.
- **Glen** practitioner, **30 min**, **all days 9–5 HST**, surfaced in the **client portal** (no public page).
- No payment at booking.

## Copy guidance

Client-facing portal text follows Glen's standing copy rules: **no em dashes, no ALL CAPS**. Warm, clear, consultative.

## Testing

- Pure/unit: generalized `create_booking(session_type=…)` writes the right `session_type` + glen-lane calendar row; `consult_is_ready` predicate; Zoom `create_meeting` payload shape (mock the HTTP); availability for `practitioner='glen'` + `GLEN_CONSULT_HOURS`.
- Route/integration (doppler): `not_ready` 403 until the flag is set; a full book flow (flag set → slot → book → Zoom link stored → confirmations fire) with Zoom mocked; `SlotTaken` 409; a security check that `/api/consult/*` is token-gated and the console flip is `X-Console-Key`-gated.
- Regression: existing EVOX tests stay green after the `session_type` generalization (EVOX rows default to `evox`).
- Go-live: verify Zoom `meeting:write` works on prod; one real consult booking with a test-eligible client end to end.
