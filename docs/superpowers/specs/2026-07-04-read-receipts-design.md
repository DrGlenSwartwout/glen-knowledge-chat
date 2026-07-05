# Read Receipts for Reports & Invoices (v1) — Design

**Date:** 2026-07-04
**Status:** Approved (brainstormed with Glen 2026-07-04)
**Repo:** deploy-chat

## Summary

Glen wants a **real read-receipt**: to know, at his end, whether a client actually opened a biofield **report** (in their portal) or an **invoice** (`/invoice/<token>`). Today reports only flip a boolean `notify_state.engaged` flag (no timestamp, no count, not per-scan) and invoices record nothing on open.

The mechanic (Glen's design): reports and invoices render **collapsed by default** — a header row, not the content. The header always shows the item's **generation date** — the report's test/scan date, the invoice's creation date — and carries a **"New"** badge until the client first opens it; after the first open, just the date remains. Clicking to expand reveals the content **and is the tracked open**. Because the signal is a deliberate expand-click (a POST), not a page load, email-security-scanner bots and the owner's own console previews don't register as opens — a "opened" genuinely means the client opened it.

## Scope

**v1 = read receipts for reports + invoices:** collapse-by-default with New/date badge, click-to-open tracking, false-open filtering, and owner-side surfacing (orders board for invoices, reveals console for reports). Ships behind `READ_RECEIPTS_ENABLED` (default OFF; flag-off = today's behavior, content shown immediately, no tracking).

**Deferred:** recording *who* opened (viewer identity beyond the report's subject), per-open event history/analytics, opens for other document types, client-facing "seen by your practitioner" indicators, notification-routing (that is its own separate sub-project).

## The mechanic

- **Collapsed by default.** A report (per scan) and an invoice render as a header card that always shows the item's **generation date** (report = its test/scan date; invoice = the date it was generated) plus a **New** badge until the client first opens it; after the first open the badge is gone, leaving just the date. The content (report body / invoice line items) is hidden until the client clicks to expand.
- **Expand-click = the open.** Expanding fires a token-scoped POST to a track-open endpoint, which records the open and flips the badge New→date. Collapsing again does nothing; a later re-expand bumps `last_opened` + `open_count`.
- **Real-receipt property:** the tracked signal is an explicit human click that issues a POST. Link-prefetching scanners issue GETs and never expand → not counted. The owner's console-authenticated previews are excluded (see False-open filtering).

## Data model

New module `dashboard/opens.py` owning one `LOG_DB` table:

```
portal_opens (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  kind          TEXT NOT NULL,     -- 'report' | 'invoice'
  key           TEXT NOT NULL,     -- report: lower(email)+'|'+scan_date ; invoice: invoice token
  first_opened  TEXT,
  last_opened   TEXT,
  open_count    INTEGER NOT NULL DEFAULT 0,
  UNIQUE(kind, key)
)
```

Functions (pure, tested): `init_opens_table(cx)`; `record_open(cx, kind, key) -> {first_opened,last_opened,open_count}` (upsert: insert with count=1 first time, else `last_opened=now`, `open_count+=1`); `get_open(cx, kind, key) -> {...}|None`; `opens_for(cx, kind, keys) -> {key: {...}}` (batch, for console lists).

`key` builder is centralized: `report_key(email, scan_date) = f"{email.strip().lower()}|{scan_date}"`, `invoice_key(token) = token`.

## Track-open endpoints (POST, token-scoped, owner-filtered)

- **`POST /api/portal/<token>/open`** `{scan_date}` — resolves the token to its portal; applies the **same `?member=` household resolution** as `api_client_portal` (so a caregiver expanding a member's report records against the *member's* key, `member_email|scan_date`); records `record_open(cx, 'report', report_key(email_for_reports, scan_date))`; returns the updated open status. Skips recording (but still returns 200) when the request is an owner preview (see below).
- **`POST /api/invoice/<token>/open`** — resolves the token to its order; records `record_open(cx, 'invoice', invoice_key(token))`; returns status. Same owner-skip.

Both are POST (never GET), so a scanner's link prefetch cannot trigger them.

## False-open filtering (explicit requirement)

1. **Bots / link scanners:** excluded structurally — the open is only recorded by an explicit expand-click issuing a POST to `/open`. Prefetchers issue GETs against the page/data endpoints, never the POST. No heuristic needed.
2. **Owner previews:** when the caller is the owner — `_portal_console_ok()` true (console key / owner token present) — the `/open` endpoint returns 200 but does **not** record. So Glen/Rae/Shaira opening a client's portal or invoice to check it never counts as a client open.
3. **Debounce:** a re-expand within a short window (5 seconds) of `last_opened` updates `last_opened` but does **not** increment `open_count` (prevents a double-click or accordion toggle from inflating the count). `first_opened` is set once and never changes.

Legitimate opens that DO count: the client on their own token, and a household caregiver expanding a linked member's report (that is a real "someone saw it," which is the point).

## Client render

- **Portal report** (`static/client-portal.html`): when `READ_RECEIPTS_ENABLED`, the report body renders inside a collapsed card whose header shows the picked `scan_date` + a **New** badge when that `(email, scan_date)` has no recorded open. Expanding calls `POST /api/portal/<token>/open` with the scan_date (carrying `?member=` if in member-view), reveals the body (still subject to the existing paywall blur), and flips New→date. The scan-date selector badges each date **New** vs plain from the payload's opens map. Flag-off → body shown expanded as today, no card, no tracking.
- **Invoice** (`static/invoice.html`): when enabled, the page renders a collapsed summary header (invoice date + New/date badge); clicking reveals the line items and calls `POST /api/invoice/<token>/open`. Flag-off → invoice shown as today.
- **Payload additions:** `api_client_portal` adds an `opens` map `{scan_date: {first_opened,last_opened,open_count}}` for the current email (all the client's scan dates, so the selector can badge). `api_invoice_get` adds `opened: {first_opened,last_opened,open_count}|null`.

## Owner surfacing

- **Invoices → orders board:** each order with an invoice shows "Opened M/D/YY (N×)" or "Not opened yet," from `portal_opens` kind=`invoice`. A batch `opens_for` lookup keyed by the orders' invoice tokens.
- **Reports → reveals console** (`/console/biofield-reveals`): each published report shows its open status (from kind=`report`, `email|scan_date`). Composes with the household work — you see *which member's* report was opened and when.
- A read-only console endpoint `GET /api/console/opens?kind=&keys=` (or folded into the existing reveals/orders payloads) supplies the data; `_portal_console_ok()`-gated.

## Data flow

Client expands a report → `POST /api/portal/<token>/open {scan_date}` → owner-check → `opens.record_open('report', email|scan_date)` → returns status → card flips New→date. Owner loads `/console/biofield-reveals` → reveals payload joins `opens.opens_for('report', [...keys])` → shows opened/when per report. Same shape for invoices via `/api/invoice/<token>/open` and the orders board.

## Components / files

- **New `dashboard/opens.py`** — `portal_opens` table + `init_opens_table`/`record_open`/`get_open`/`opens_for` + `report_key`/`invoice_key`.
- **`app.py`** — `POST /api/portal/<token>/open`, `POST /api/invoice/<token>/open`; `opens` in `api_client_portal` payload; `opened` in `api_invoice_get`; open-status join in the reveals + orders console payloads; a `_read_receipts_enabled()` flag helper.
- **`static/client-portal.html`** — collapsed report card + expand-tracks + selector New badges.
- **`static/invoice.html`** — collapsed invoice + expand-tracks.
- **Console** — reveals page + orders board show open status (existing pages, additive).

No change to `portal_biofield_reports`, the invoice/order schema, or the reveal/publish pipeline.

## Error handling

- Track-open is best-effort: a record failure never breaks the expand (the content still shows); the endpoint returns a soft error but the UI proceeds.
- `READ_RECEIPTS_ENABLED` off → no collapse, no `opens`/`opened` payload keys, `/open` endpoints inert (or absent) → today's behavior byte-for-byte.
- Owner-preview and debounce paths return 200 without inflating counts.
- Missing `portal_opens` table / any lookup failure → console shows "unknown"/blank, never errors.

## Testing

- **`opens.py` units:** `record_open` first call sets first=last, count=1; second call bumps last + count; debounce window suppresses the count increment but updates last; `get_open` None when absent; `opens_for` batch.
- **Report route:** `POST /api/portal/<token>/open` records against `email|scan_date`; owner (console key) call does NOT record; member-view (`?member=`) records against the member's key; flag-off inert.
- **Invoice route:** `POST /api/invoice/<token>/open` records against the token; owner call does not; flag-off inert.
- **Payload:** `api_client_portal` carries `opens` (per scan date) only when enabled; `api_invoice_get` carries `opened`.
- **Console:** reveals/orders payloads surface open status; `_portal_console_ok`-gated read endpoint.
- **Render (static):** collapsed card renders New vs date from payload; expand issues the POST; escaped; flag-off shows content as today.

## Out of scope / future

- Recording viewer identity (opened-by-whom beyond the report's subject key).
- Per-open event log / analytics / opened-from-where.
- Client-facing "seen by your practitioner" indicators.
- Notification-routing (member emails → caregiver and/or self) — separate sub-project.
- Read receipts for other document types (protocols, emails).
