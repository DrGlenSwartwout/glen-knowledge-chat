# Coaching — request + accept + waitlist + upsell offer (coaching arc, slice 2) — Design

**Date:** 2026-07-05
**Status:** Approved in brainstorm with Glen 2026-07-05 (free student request/accept + capacity + waitlist + upsell OFFER card; recurring paid checkout is slice 2b).
**Repo:** deploy-chat

**Relates to / reuses:**
- Slice 1: `dashboard/coach_directory.py` (`coach_volunteers`, `list_active`, `get_volunteer`, `_lc`), `GET /api/community/coaches` (member active-coaching-window gated), the practitioner "Coaching volunteer" control, `_practitioner_session_pid()` + `practitioner_email_by_id` (coach auth), `_evox_ident` (member auth), `coaching.active_window` (member eligibility).
- Existing products referenced by the upsell copy: EVOX session (live), **Causal Biofield Analysis** ($300 service; `_BIOFIELD_ITEM_NAME`). Glen notification via the existing email/console rails.
- `LOG_DB`, `_db_lock`, sqlite conventions, `CONSOLE_SECRET`.
- [[project_pb_to_illtowell_evox]], [[project_membership_prepay_ladder]] (recurring billing lives here — used by slice 2b, not this slice).

## Context and boundary

Coaching arc: slice 1 (coach directory) LIVE. **Slice 2 (THIS) = the pairing:** a member requests a coach, the student accepts up to their capacity, a waitlist catches the overflow, and an upsell offer presents paid coaching when the free students are full. Slice 3 (deferred) = the 1:1 coaching thread (with report/block) that a formed pairing opens on.

**Free vs paid:** the certification-student volunteer coaches are **free**. This slice does NOT charge anyone. The paid tiers (Rae $100/mo including an EVOX session; Dr. Glen $200/mo including a Causal Biofield Analysis) appear here only as an **offer card** with an "I'm interested" action that flags Glen — the recurring subscription + entitlement bundling is **slice 2b**.

**Privacy line (carried from the arc):** no email is exposed either direction. A member requesting a coach consents to that one coach seeing their **first name + focus** (nothing else); a coach is referenced by an opaque ref, never their email. The actual conversation waits for slice 3.

## Scope

**Member requests a free student coach (by opaque ref) → student accepts up to capacity → waitlist when all full → upsell offer to paid coaching.** New relationship + waitlist + interest stores, the request/accept/waitlist/interest routes, and the two surfaces (member card, practitioner requests list).

**Deferred:** slice 2b (recurring paid subscription + EVOX/Biofield bundling), slice 3 (the 1:1 thread + report/block), coach-to-member messaging, waitlist auto-assignment beyond a simple next-in-line offer.

## Components

### 1. Connect store (`dashboard/coach_connect.py`)

- `coach_requests(id INTEGER PK, coach_email TEXT, member_email TEXT, member_name TEXT, note TEXT, status TEXT, created_at TEXT, decided_at TEXT, UNIQUE(coach_email, member_email))` — status ∈ {`pending`, `accepted`, `declined`}. This is the **double opt-in**: the member APPLIES (creates the pending row + a short `note` on what they are working on), the coach ACCEPTS (both sides consent). `member_name` (first name) + `note` captured at apply time.
- `coach_waitlist(member_email TEXT PRIMARY KEY, created_at TEXT)` — members waiting when all coaches are full.
- `coaching_interest(id INTEGER PK, member_email TEXT, tier TEXT, created_at TEXT, UNIQUE(member_email, tier))` — `tier` ∈ {`rae`, `glen`}; a member flagging interest in the paid upsell (no charge).
- Functions (pure sqlite, emails lowercased, no app imports): `init_connect_tables(cx)`; `coach_ref(email) -> str` (stable opaque ref = `sha256(lower(email))[:16]`, and `email_for_ref(cx, ref) -> email|None` resolving a ref against the active volunteer roster); `create_request(cx, coach_email, member_email, member_name, note) -> int` (idempotent per pair; no-op if the member already has an active request — see cap); `member_active_request(cx, member_email) -> dict|None` (their one pending/accepted request); `requests_for_coach(cx, coach_email, status="pending") -> [dict]`; `accepted_count(cx, coach_email) -> int`; `set_request_status(cx, request_id, status)`; `join_waitlist(cx, member_email)`; `on_waitlist(cx, member_email) -> bool`; `record_interest(cx, member_email, tier)`.

### 2. Coach ref + full-coach exclusion (`dashboard/coach_directory.py`)

- `list_active` gains a `ref` per coach (`coach_connect.coach_ref(email)`) so a member can request a specific coach without seeing the email, and **excludes coaches whose `accepted_count >= capacity`** (full coaches take no new requests). Signature/return still member-safe: `{ref, name, focus, intro_video_url}` (still NO email). The full-exclusion needs the accepted count, so `list_active` takes the connect store into account (or the route composes it) — keep `list_active` returning candidates and let the route drop full ones, whichever keeps the store modules cleanly separated.

### 3. Member routes (`app.py`, member portal-token + active coaching window)

- `GET /api/community/coaches` (extended): returns `{eligible, coaches:[{ref,name,focus,intro_video_url}], my_request: {ref, status}|null, all_full: bool}`. `all_full` = true when cert-ok volunteer coaches exist but every one is at capacity (→ frontend shows waitlist + upsell). `my_request` is the member's current request status (so the card shows "requested"/"accepted").
- `POST /api/community/coach-request {coach_ref, note}` → resolve ref → coach; **one active request per member** (409 `already_requested` if they already have a pending/accepted one); create the pending application with the member's `note` (trimmed, capped ~500 chars). Returns `{ok, status:"pending"}`.
- `POST /api/community/coach-waitlist` → `join_waitlist`; returns `{ok}`. (Meaningful when `all_full`.)
- `POST /api/community/coaching-interest {tier}` → `tier in {rae, glen}` (else 400); `record_interest`; best-effort notify Glen (email/console) that a member is interested in the paid tier; returns `{ok}`. NO charge.

### 4. Coach routes (`app.py`, practitioner session)

- `GET /api/practitioner/coach-requests?token=…` → `{pending:[{request_id, member_name, note}], coachees:[{member_name}], capacity, slots_left}` — the coach's pending applications (member **first name** + their **note**, never email), their accepted coachees, and how many slots remain (`capacity - accepted_count`).
- `POST /api/practitioner/coach-request/respond {request_id, accept}` → verify the coach owns the request (else 404); on `accept`: refuse with 409 `at_capacity` if `accepted_count >= capacity`, else `set_request_status("accepted")`; on decline: `set_request_status("declined")`. Returns `{ok, status}`.

### 5. Surfaces

- **Member "Meet your coaches" card** (`static/client-portal.html`): an "Apply to this coach" action per coach card that captures a short note ("what are you working on") and posts `{coach_ref, note}`; shows the member's application status when they have one; when `all_full`, show a "Join the waitlist" action **and** the upsell offer card: "Our student coaches are full right now. Join the waitlist, or start coaching now with Rae ($100/mo, includes an EVOX session) or Dr. Glen ($200/mo, includes a Causal Biofield Analysis)." Two "I'm interested" buttons posting `coaching-interest` with `tier=rae`/`glen`. Copy: no em dashes, no ALL CAPS.
- **Practitioner "Coaching volunteer" control** (`static/practitioner-portal.html`): an applications list — each pending application shows the member first name + their note with Accept / Decline, and a "slots left" count; accepted coachees listed below. Member name/note via `textContent` (no injection).

## Config

No new required env. The paid tier prices ($100/$200) are display-only copy in this slice (real billing is slice 2b). Optional `COACHING_INTEREST_NOTIFY_EMAIL` (defaults to Glen's console email) for the interest notification.

## Privacy

- Coaches are referenced by an opaque `ref` (sha256 of email, truncated); the email is never sent to a member. A member is shown to a requested coach as **first name + focus only** (no email), and only because the member initiated the request (consent). The waitlist and interest stores hold the member's own email keyed to themselves; no cross-member exposure.

## Testing

- Pure/sqlite (`dashboard/coach_connect.py`): `coach_ref` stable + `email_for_ref` round-trips against active volunteers and returns None for unknown/inactive; `create_request` one-active-per-member cap (second request while pending → no new row / rejected); `set_request_status` → `accepted_count` reflects accepted only; `requests_for_coach` returns pending with member_name + note, NO email; `join_waitlist`/`on_waitlist`; `record_interest` idempotent per (member, tier).
- Route/api: member request (bad ref → 404; no active window → ineligible; one-active cap → 409); `GET coaches` excludes full coaches + sets `all_full` when all at capacity + returns `my_request`; waitlist join; `coaching-interest` records + best-effort notify (bad tier → 400). Coach: `coach-requests` returns pending (first name only, NO email) + slots_left; `respond` accept within capacity → accepted, at capacity → 409, non-owner → 404. Privacy assertions: no coach email in any member payload; no member email in any coach payload.
- Regression: slice-1 directory + signup untouched except the additive `ref` + full-exclusion; Community A/B/C untouched.
- Go-live: two members request a coach with capacity 1 → coach accepts one (slot fills, they drop from the directory), the second member sees the waitlist + upsell; a member taps "I'm interested" in Glen's tier → Glen is notified.

## Deferred (coaching arc, later slices)

- **Slice 2b:** the recurring paid-coaching subscription (Rae $100/mo + EVOX, Glen $200/mo + Causal Biofield), reusing the membership/billing + EVOX + Biofield systems, bundling the included service each cycle.
- **Slice 3:** the 1:1 coaching thread on an accepted pairing, with report/block + moderation (shared channel with the peer-matching arc).
- Waitlist auto-assignment/notification when a slot frees (this slice stores the waitlist; working it is manual/next).
