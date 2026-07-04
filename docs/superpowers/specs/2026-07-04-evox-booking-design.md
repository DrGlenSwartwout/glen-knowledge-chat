# EVOX Booking — bring Rae's EVOX scheduling home (PB→illtowell slice 1) — Design

**Date:** 2026-07-04
**Status:** Approved in brainstorm with Glen 2026-07-04; **two feasibility deviations flagged below for Glen to confirm on review** (see “Changed assumptions”).
**Repo:** deploy-chat
**Session:** PB→illtowell · EVOX

**Relates to:**
- Parent program: **PB→illtowell unification** — stop making clients live in both Practice Better and illtowell.com; bring PB's client-facing capabilities home to the portal. This spec is **slice 1 of the appointment loop**.
- Calendar subsystem: `calendar_events` table + `get_calendar`/`post_calendar` in `app.py` (~18854–19144), owner map / `rae` lane (`project_console_calendar`, PR #231/#343).
- Portal identity: `dashboard/portal_identity.py:resolve_identity`, `dashboard/customers.py:find_or_create_by_email`, `dashboard/client_portal.py:upsert_portal`.
- Commerce / money: `data/products.json`, in-house order builder `/api/orders/manual`, `dashboard/orders.py:set_order_payment` (record-payment), Stripe checkout.
- Products/SKU pattern: `info_only`/manual-order overrides (`reference_member_gated_qty_pricing`, dispensary spec 2026-07-03).

---

## Summary

Bring **EVOX session booking** (Rae's ZYTO-based sessions) home from Practice Better to the illtowell.com client portal. The EVOX *scan* still runs in ZYTO; what moves is everything PB currently wraps around it — a **self-attest readiness onboarding**, **slot booking against Rae's availability**, the **connection handoff**, and **reminders/confirmation**. Rae stops juggling PB-plus-ZYTO; the client stays on illtowell.com.

This is the thin end-to-end vertical that proves the whole appointment-loop spine (a session-type catalog, gating, availability + booking, a per-type connection handoff, and the in-house money rails) on the single session type where the unification payoff is highest for Rae.

## Scope

**v1 = EVOX only**, one practitioner (Rae), medium = **phone + internet (no Zoom)**.

Built as **one booking engine reading a small session-type catalog**, so the four later types (Triage/Discovery, New-member onboarding, Biofield Consult, MasterClass) bolt on as data + small deltas — but only EVOX ships in v1.

**Deferred to their own specs (explicitly out of scope):** Zoom/video handoff; the other four session types; the one-to-many MasterClass *event* shape (v1 is 1:1 slots only); client-facing intake *forms*; any admin UI to author session types; live Google free/busy and two-way Google Calendar write API; a full client-tagging UI.

---

## Changed assumptions (feasibility — Glen to confirm on review)

During grounding, two things I described in the brainstorm turned out **not** to exist in the codebase. The design adapts; flagging so Glen can confirm the substitutions:

1. **No live Google free/busy read, no Google Calendar write API.** Only Gmail uses `googleapiclient`; there is no calendar-write or free/busy path. **Adaptation:** availability is computed from Rae's **defined EVOX office-hours window** minus (a) EVOX slots already booked in *our* system and (b) busy rows already present in the local **`calendar_events`** table — which mirrors Rae's Google calendar via the existing hourly sync that feeds the console team calendar (`rae` lane). Write-back to Rae's + the client's real calendars is done with an **ICS invite email** (a standard `.ics` attachment; accepting it lands the event on any calendar app — no Google API needed).
   - **Sync directionality (confirmed with Glen):** Google→internal is a one-way hourly read (Rae's Google events → `calendar_events`), **cadence tunable** (can tighten to ~15 min to shrink the collision window). Internal→Google is **not** a reverse hourly sync — each booking's **ICS invite lands on Rae's + the client's real calendars instantly on accept**, no Google write API.
   - **Known limitation:** availability is only as fresh as the last read sync. A conflict Rae adds to Google within that window could theoretically collide. Mitigation: office-hours windows are narrow and Rae-controlled; every booking emails Rae immediately; tighten the read cadence and/or add live free/busy in a later slice if it proves painful.

2. **The hand cradle is not in `data/products.json`.** Searched cradle/ZYTO/EVOX/$297 — every hit is an infoceutical. **Adaptation:** adding the hand-cradle SKU ($297 + shipping) is a task in the plan, not an assumed existing product.

---

## Session-type catalog (the seam)

A declarative catalog in code (not an admin UI). Each entry is a plain dict; the booking engine reads these fields. v1 ships exactly one entry (EVOX); the others are documented here as the target shape but **not built**.

```
SESSION_TYPES = {
  "evox": {
    "label": "EVOX Session",
    "practitioner": "rae",          # owner lane / whose availability
    "duration_min": 60,
    "medium": "phone",              # phone | video(future) | group(future)
    "shape": "slot",                # slot | event(future)
    "gate": "readiness:evox",       # readiness | member | entitlement | invite | public
    "payment": "invoice_after",     # invoice_after (default) | prepay_optional
    "prepay_sku": "evox-session",   # single/multi prepay product (optional purchase)
  },
  # future (NOT built in v1): "triage","onboarding","biofield-consult","masterclass"
}
```

Gate semantics are pluggable: v1 implements only `readiness:evox`. `duration_min`, `practitioner`, `medium` drive slot generation, calendar write, and the confirmation template respectively.

---

## Components

### 1. EVOX Setup surface + entry (identity first)

- A portal surface **“EVOX Setup”** inside `/portal` (client-portal.html), plus a **public `/evox` start page** so *anyone* can begin (Glen: eligibility = anyone).
- Public start captures **name + email** → `customers.find_or_create_by_email` + `client_portal.upsert_portal` to mint a person + portal (identity is required to hold checklist state and attach a booking). Returns them into the checklist via their portal.
- Existing clients see “EVOX Setup” in their portal directly (identity already resolved via `resolve_identity`).

### 2. Readiness gate — self-attest checklist (no human approval)

Per-person state in a new table `evox_readiness`. Four items, all self-attested:

1. ☐ Windows 10/11 PC (have or can access)
2. ☐ Hand cradle — **Buy ($297 + shipping)** *or* “I already have access”
3. ☐ Headset + microphone
4. ☐ ZYTO software installed **and** setup verified with ZYTO support (M–Th)

- **Buy** runs the `hand-cradle` SKU through the existing checkout inline; a completed purchase auto-checks item 2 (attribute the order → flip the box). “Already have access” is a one-click attest.
- When all four are true → `readiness_complete = true` → booking unlocks. No Rae click.
- Tag hook: on completion, tag person `evox-ready` (see Tags).

### 3. Availability + 1:1 slot booking (Rae)

- **Office hours:** a small config for Rae's bookable EVOX window (e.g. `EVOX_HOURS = Mon–Thu 09:00–16:00 HST`, 60-min grid). Config, not a UI, in v1.
- **Availability = office-hours grid − our booked EVOX slots − busy rows in `calendar_events` for the `rae` lane** (windowed to the requested range; reuse `_calendar_range_window`).
- Client picks an open slot → create a booking row (`evox_bookings`) + insert a `calendar_events` row on the **`rae` lane** (synthetic id, so it renders on the console team calendar exactly like a delegated event) → send:
  - client confirmation email **with an `.ics` invite** (time + prep instructions),
  - Rae notification email **with the same `.ics`** (lands it on her real calendar on accept).
- Double-book guard: unique constraint on `(practitioner, start_ts)` for active bookings.

### 4. Connection handoff + confirmation (phone + internet, no Zoom)

- Medium = phone. **No video link is generated.**
- Confirmation content (client): appointment time (their tz), and prep — “Have your Windows PC on with the ZYTO software open, hand cradle connected, headset ready. **At your appointment time, call Rae at <number>.**” (Call direction settled: **client calls Rae** — Rae's number is in the confirmation + ICS.)
- Reminder email 24h before (reuse existing scheduled-email rails if present; else a daily cron pass over `evox_bookings`).

### 5. Money

- **Default = invoice at session time.** Booking never blocks on payment. After the session Rae records payment on the existing in-house rails: create/settle an in-house order line for the EVOX session and `set_order_payment(method, amount_cents)` (check/cash/etc.), same flow as the Karin/biofield-consult pattern.
- **Optional prepay** — a **`evox-session` SKU** (single) plus a multi-session package, sold on-site. A prepay purchase credits a **session balance** (`evox_session_credits`, integer, per person). At booking, if the person has credit > 0, the booking marks `prepaid=true` and decrements one; otherwise `prepaid=false` → Rae invoices after.
- **Hand cradle** — **new SKU** `hand-cradle`, $297 + shipping (shipped via the normal packer / flat-rate; not `info_only`). Purchasable from the checklist and standalone.

### 6. Tags (woven in, minimal)

- A minimal person-tag store (verify whether one already exists before adding; else a tiny `person_tags(email, tag)` table). Two writes only in v1:
  - readiness complete → `evox-ready`
  - first booking → `evox-client`
- No tag-management UI in v1 (that's the Tags sub-project). These are segmentation seeds for later.

---

## Data model changes

- `evox_readiness(email TEXT PK, pc_ok, cradle_ok, cradle_source TEXT /* buy|access */, headset_ok, zyto_ok, completed_at, updated_at)` — lazy `CREATE TABLE IF NOT EXISTS`.
- `evox_bookings(id, email, practitioner TEXT, start_ts, end_ts, status /* booked|cancelled|done */, prepaid BOOL, calendar_event_id, ics_uid, created_at)` — unique active `(practitioner, start_ts)`.
- `evox_session_credits(email TEXT PK, credits INT)` — prepay balance.
- `person_tags(email TEXT, tag TEXT, created_at)` **or** the existing tag store if one is found (plan verifies first).
- `data/products.json`: add `hand-cradle` ($29700 + shipping) and `evox-session` (prepay) SKUs.
- No change to `calendar_events` schema — bookings insert an ordinary `rae`-lane row (synthetic id like the delegate-move pattern).

## Integration points

1. **Public `/evox`** + portal “EVOX Setup” panel (client-portal.html) → checklist API.
2. **Checklist API** (`/api/evox/readiness` GET/POST attestations; buy → existing checkout, order→attest attribution).
3. **Availability API** (`/api/evox/availability?range=` → open slots from office-hours − calendar_events busy − booked).
4. **Booking API** (`/api/evox/book` → row + `calendar_events` insert + two `.ics` emails + tag).
5. **Money**: hand-cradle & evox-session SKUs on existing checkout; post-session invoice on `/api/orders/manual` + `set_order_payment`.
6. **Reminders**: 24h pre-session email (existing scheduled-email rail or a daily cron).

## Settled decisions

- Self-attest readiness, **no human gate** (Glen).
- Eligibility = **anyone** (public entry), identity minted on start.
- Medium = **phone + internet, no Zoom** in v1 (Glen). **Call direction: client calls Rae** (Glen).
- Sync: Google→internal hourly read (**cadence tunable**); write-back via **instant ICS-on-accept**, not a reverse sync (Glen confirmed).
- Payment: **invoice-after by default**, prepay single/multi **optional**; booking never blocks on money (Glen).
- Hand cradle **sold on-site**, $297 + shipping (Glen); **must be added** (not in catalog).
- Availability from **local `calendar_events` + office-hours**, write-back via **ICS invite** (adaptation — no Google write API).

## Out of scope (restate)

Zoom/video; other 4 session types; MasterClass event shape; client intake forms; session-type admin UI; live Google free/busy + two-way Google write; tagging UI.

## Testing

- Pure unit: availability computation (office-hours grid − busy − booked, tz-correct), readiness→unlock predicate, prepay decrement, `.ics` generation (valid VEVENT, correct UID/DTSTART/DURATION).
- Route/integration through the real entry points (avoid mock-masked-green: assert `calendar_events` actually gets a `rae` row and the SKU order actually flips the cradle box) — per `feedback_mock_masked_green_tests`.
- Manual go-live: one real EVOX booking end-to-end with Rae (checklist → book → both `.ics` arrive → console calendar shows the rae-lane event → invoice-after).
