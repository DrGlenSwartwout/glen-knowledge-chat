# Console Calendar — Design Spec

**Date:** 2026-06-22
**Status:** Increment 1 specced; Increments 2 & 3 noted as follow-on specs.
**Flag:** `CALENDAR_ENABLED` (console-only, default-off)

---

## Goal

A team calendar inside the console showing the daily schedule for **Glen**, **Rae**, and
**Shaira**, with role-based visibility, multiple view ranges, two entry kinds
(events vs accomplishments), and audio alarms before scheduled times.

## Roles → people (existing RBAC, reused as-is)

| Role | Person | Sees |
|------|--------|------|
| `owner` | Glen | all three people's entries |
| `ops` | Rae | all three people's entries |
| `va` | Shaira | **only** `person='shaira'` entries |

This is the visibility rule Glen specified ("Rae's & Glen's schedules seen by Rae and
Glen; Shaira's seen by all"), expressed as: **owners see everything; the VA sees only her
own lane.** Enforced **server-side** in the list endpoint — the client never receives
entries it isn't allowed to see (matches the codebase's server-withheld anti-leak pattern).

## Time zone

"One shared zone (HST)." Times stored in **UTC**, rendered in **HST** (`Pacific/Honolulu`)
for every viewer. Alarms compute against the UTC instant.

---

## Increment decomposition

| # | Scope | Status |
|---|-------|--------|
| **1. Core calendar** | table + console page + 4 view ranges + event/accomplishment toggle + manual CRUD via dispatch + server-side visibility + **in-browser** audio alarm | **this spec** |
| **2. Google Calendar sync** | local cron reads `token-calendar.json` (`calendar.readonly`), walks a **calendar→person map**, upserts events (idempotent by `source_id`) | follow-on spec |
| **3. Phone-push alarms** | Web Push (service worker + VAPID) + server cron firing upcoming alarms with no tab open | follow-on spec |

---

## INCREMENT 1 — Core calendar (this spec)

### Data model — new table `calendar_entries`

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | |
| `person` | TEXT | `'glen' \| 'rae' \| 'shaira'` — whose lane |
| `kind` | TEXT | `'event' \| 'accomplishment'` |
| `title` | TEXT | required |
| `start_utc` | TEXT (ISO-8601 Z) | required; the alarm/sort anchor |
| `end_utc` | TEXT (ISO-8601 Z) | nullable |
| `all_day` | INTEGER (0/1) | default 0 |
| `notes` | TEXT | nullable |
| `alarm_minutes` | INTEGER | nullable; minutes-before-start to chime |
| `source` | TEXT | `'manual'` (Inc. 1). `'gcal'` reserved for Inc. 2 |
| `source_id` | TEXT | nullable; gcal event id for Inc. 2 upsert idempotency |
| `created_by` | TEXT | actor name/role |
| `created_at` | TEXT (ISO-8601 Z) | |
| `updated_at` | TEXT (ISO-8601 Z) | |

Index on `(start_utc)` and `(person, start_utc)`. Table created by an
`_init_calendar_table()` helper called at startup, mirroring `_init_todos_table()`.

### Server-side visibility filter (pure, unit-tested)

```
visible_persons(role):
  owner / ops -> ['glen','rae','shaira']
  va          -> ['shaira']
  else        -> []
```

The list endpoint always intersects the query with `visible_persons(actor.role)`. A `va`
request for `person=glen` returns empty, never an error that leaks existence.

### Read endpoint

`GET /api/console/calendar?range=<day|2day|week|month>&date=<YYYY-MM-DD>&kind=<event|accomplishment|all>`

- Auth: `CONSOLE_SECRET` / `X-Console-Key` (same gate as the other console APIs).
- `range` → a `[start_utc, end_utc)` window anchored on `date` (default today, HST):
  - `day` = the HST day of `date`
  - `2day` = `date` + next day
  - `week` = the HST week containing `date` (Mon–Sun)
  - `month` = the HST calendar month containing `date`
- Filters by `visible_persons(role)` and (if given) `kind`.
- Returns entries grouped/sortable by `person` and `start_utc`, each carrying an
  HST-rendered display time computed server-side (client also has the UTC instant for the
  alarm timer).

### Write actions (dispatch spine — `dashboard/calendar_actions.py`)

All writes go through `dispatch_action` exactly like `tasks.complete_todo`.

| Action key | Risk tier | Permission | Effect |
|------------|-----------|------------|--------|
| `calendar.add_entry` | `LOW_WRITE` | owner, ops, va\* | insert a `manual` entry |
| `calendar.update_entry` | `LOW_WRITE` | owner, ops, va\* | update by id |
| `calendar.delete_entry` | `LOW_WRITE` | owner, ops, va\* | delete by id |
| `calendar.complete` | `LOW_WRITE` | owner, ops, va\* | flip an `event` → `accomplishment` (records `updated_at`) |

\* **VA scoping:** a `va` actor may only add/update/delete/complete entries where
`person='shaira'`. The executor checks the target `person` (for add) or the row's existing
`person` (for update/delete/complete) and returns a permission error otherwise. Owners
(Glen/Rae) may write any lane — including scheduling work into Shaira's lane.

### Console page — `static/console-calendar.html` + route `/console/calendar`

- **View-mode switch:** Today (1d) · Today+Tomorrow (2d) · This Week · This Month.
- **Content toggle:** Events ⇄ Accomplishments (re-queries with `kind`).
- **Per-person color lanes** for Glen / Rae / Shaira. A `va` session renders only the
  Shaira lane (because the server only returns Shaira entries for `va`).
- **Add/Edit form:** who (person) · kind · title · date+time (HST input, converted to UTC
  on submit) · optional `alarm_minutes` ("N min before") · notes.
- **"✓ Did it"** button on an event → calls `calendar.complete`.
- All writes POST to the existing `/api/action/<key>` path with `X-Console-Key`.
- Add the page to `static/console-search-index.json` (hand-maintained Pages search).

### In-browser audio alarm (Inc. 1)

- The open page keeps the next-N upcoming entries (with `alarm_minutes` set) and runs a
  client timer. At `start_utc − alarm_minutes` it: plays a **Web Audio** chime, flashes the
  entry visually, and (if the user granted permission) raises a **Notification**.
- **Tab-open only.** No sound when no tab is open — that capability is **Increment 3**
  (phone push). The UI states this near the alarm control so expectations are clear.
- Each alarm fires at most once per page-session (track fired ids in memory).

### Testing (Increment 1)

- `visible_persons` — owner/ops see all three; va sees only shaira; unknown sees none.
- Range→window math for `day / 2day / week / month` around an HST anchor (incl. month
  boundary and HST↔UTC day-shift).
- Each dispatch action's permission gating, including **va cross-lane denial** (va cannot
  write a `person='glen'` entry; va add defaults/forces `person='shaira'`).
- List endpoint applies the visibility filter (va request for glen → empty).
- Mirrors the existing console test suites (`test_*` around dispatch + console APIs).

### Out of scope for Increment 1 (explicitly deferred)

- Google Calendar sync (Increment 2).
- Phone / push-when-tab-closed alarms (Increment 3).
- Recurring-event rules (RRULE) — manual entries are single instances in Inc. 1; recurrence
  arrives naturally via gcal sync in Inc. 2.
- Drag-to-reschedule / week-grid pixel layout — Inc. 1 is a readable list per range, not a
  pixel-grid calendar. (Can be a later visual pass.)

---

## INCREMENT 2 — Google Calendar sync (follow-on spec, summary only)

- **Calendar→person map** (config: env JSON or a tiny table) — Glen's **2** calendars →
  `glen`, Rae's → `rae`, Shaira's → `shaira`. Mapping is the source of truth; one person
  can have many calendars.
- **Auth path:** Rae and Shaira **share their calendars into Glen's Google account**
  ("See all event details"). Glen's existing `token-calendar.json` (`calendar.readonly`)
  then reads **all** mapped calendars via `calendarList` — **no new OAuth tokens**.
- **Cron:** extends the `console_push_cron.py` pattern (local Mac cron → POST to console
  with `X-Console-Key`). New endpoint `POST /api/console/calendar/sync` upserts each event
  as `source='gcal'`, `source_id=<event id>`, `person=<mapped>`, idempotently (re-runs
  update in place; deletes that vanish from gcal are tombstoned/removed).
- Manual and gcal entries coexist in the same table/lanes; gcal entries are read-only in
  the console (edit them in Google).

## INCREMENT 3 — Phone-push alarms (follow-on spec, summary only)

- Web Push: service worker + VAPID keys; each person subscribes once per device
  (iOS requires "Add to Home Screen").
- A server cron scans entries with `alarm_minutes` and an upcoming `start_utc`, and pushes
  to that person's subscriptions at `start − alarm_minutes` — fires with **no tab open**.
- Net-new infra (no web-push/SMS/service-worker exists today), which is why it is built
  last, on top of a proven calendar.

---

## Rollout

Increment 1 ships behind `CALENDAR_ENABLED` (console-only, default-off), flipped in Doppler
`remedy-match/prd` when the visual pass is done — consistent with how every other console
feature in this codebase rolls out.
