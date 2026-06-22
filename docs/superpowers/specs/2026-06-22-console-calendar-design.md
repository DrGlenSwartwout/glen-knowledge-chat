# Console Calendar — Design Spec (as built)

**Date:** 2026-06-22
**Status:** Increment 1 built (this PR). Increments 2 & 3 = follow-ons.

> **Supersedes the original draft of this spec**, which proposed a new `calendar_entries`
> table. Exploration showed a calendar subsystem already exists, so Increment 1 **enhances**
> it rather than building greenfield.

## Goal

A console calendar showing the daily schedule for **Glen, Rae, and Shaira**, with role-based
visibility (Glen & Rae see all three; Shaira sees only her own), four view ranges
(Today / Today+Tomorrow / This Week / This Month), an **Events ⇄ Accomplishments** toggle, and
audio alarms a few minutes before a scheduled time.

## What already existed (reused)

- **`calendar_events`** table (`_init_calendar_table`, app.py) — Google-events mirror with
  `owner`, `cal_alert`, `UNIQUE(google_cal_id, google_event_id)`.
- **Google-Calendar sync** — local `console_push_cron.py:push_calendar_events()` walks Glen's
  `calendarList` and POSTs to `POST /api/calendar`. It already sets `owner` via a
  payment→rae/else→glen heuristic and duplicates payment events to a `_rae` copy.
- **Identity/scoping** — `_auth()`: `CONSOLE_SECRET` ⇒ `admin`; a `workspace:<owner>` access
  token ⇒ scoped. `_can_access_owner` / `_owner_from_scope` enforce "admin all, scoped own."
  Shaira already has a scoped page at `/workspace/shaira` (`shaira-workspace.html`).
- **Audio alarm** — `console.html` `playChime()` (Web Audio + spoken "starting in 3 minutes" +
  Notification + iCloud-Reminder sync) fired at `start − 3 min`.

## Built in Increment 1

### Backend (app.py)
- `_init_calendar_table()` also creates **`calendar_owner_map`** (`google_cal_id → owner`,
  seeded from the `CALENDAR_OWNER_MAP` env JSON) and **`calendar_accomplishments`**
  (`owner,title,at,notes,created_by,created_at`).
- Pure helpers: `_calendar_range_window(range, anchor)` (HST `today|2day|week|month` →
  inclusive date window) and `_calendar_owner_for(cal_id, name, map)`.
- **`GET /api/calendar`** reworked: now `_auth`-gated; admin → all three owners (or `owners=`
  subset), scoped → forced to own lane; optional `range`+`date` window (legacy "today onward"
  preserved when `range` absent); `kind=events|accomplishments`; each row tagged with `kind`.
- **`POST /api/calendar`** owner precedence: **explicit `calendar_owner_map` mapping** wins
  over the cron's heuristic `owner`, which wins over `glen`. (So Rae's/Shaira's *shared*
  calendars land in their own lane; Glen's own calendars keep the payment heuristic.)
- **`POST /api/calendar/accomplishment`** — `_auth`-gated; scoped user may only write her own
  owner (`_can_access_owner` → 403 otherwise).

### Frontend
- `console.html` `#cal-section` expanded **in place**: range switch + Events/Accomplishments
  toggle + **per-owner lanes** (Glen/Rae/Shaira, color-coded), reusing the existing alarm
  engine; a "+ Done" control logs an accomplishment for the selected person.
- `shaira-workspace.html` gains a **Calendar** section (her scoped lane) with the same controls
  and a local-tab alarm toggle.
- `static/cal-alarm.js` — shared alarm engine (chime + speech + Notification) used by the
  workspace page.

### Tests — `tests/test_calendar.py` (15)
Range-window math; owner mapping (pure + POST precedence); scoped GET (admin all / shaira own /
unauth 401); range filtering; accomplishment add + scope enforcement; default-kind exclusion.

### Visibility mapping (the requirement)
`admin` (Glen/Rae via `CONSOLE_SECRET`) → all three lanes. `workspace:shaira` token → only
Shaira. Enforced server-side in the GET (no client trust).

## Operational
- Email Shaira (laithylle.mangubat@gmail.com) the "share your Google Calendar with
  drglenswartwout@gmail.com (See all event details)" steps.
- After Rae & Shaira share their calendars in, read their `google_cal_id`s and set
  `CALENDAR_OWNER_MAP` in Doppler `remedy-match/prd`.

## Follow-on increments (not in this PR)
- **2. Multi-calendar owner mapping rollout** — mostly done server-side; remaining work is
  capturing the real `google_cal_id`s and setting the env map.
- **3. Phone-push alarms** — Web Push (service worker + VAPID) + a server cron firing alarms
  with no tab open. Net-new infra.

## Rollout
Additive, low-risk (no destructive schema change; GET stays backward-compatible when `range`
is absent). Ships with the front-end together. No new flag required; `CALENDAR_OWNER_MAP` is
the only new config and is optional (absent → legacy glen/rae behavior).
