# Available-Scan List (Sub-project A) — Design

**Date:** 2026-07-04
**Status:** Approved (brainstormed with Glen 2026-07-04)
**Repo:** deploy-chat (+ a local sync script in `~/AI-Training/02 Skills/`)

## Summary

Every client's E4L scans should be visible in their portal as a **scan history**, even the ones never analyzed — so a client (or their household caregiver) sees every scan date and knows an analysis is available to request. Today the portal only shows *published reports*; e.g. Karin has ~19 E4L scans (latest 6/28) but only one published to her portal.

This is **sub-project A**: surface the scan **dates** (read-only). Each date shows as **processed** (a published report exists — the existing openable card) or **available** (not yet analyzed — an informational row). The "request further analysis" action + the 1/mo-free / unlimited-paid rate gate are **sub-project B** (separate spec).

## The core constraint

`e4l.db` lives on Glen's Mac (`~/AI-Training/e4l.db`, kept fresh by the ingest cron); the deploy-chat portal runs on Render and **cannot read it**. So the scan-date manifest must be **synced from local → prod** into a table the portal reads. `e4l_scans` is keyed by `client_id`, not email — the email comes from `e4l_clients` (a client may have several emails).

## Scope

**v1 = sync scan-date manifests to prod + list them in the portal (read-only).** Behind `SCAN_LIST_ENABLED` (default OFF). Deferred to B: the request action, rate gate, and fulfillment/synthesis. Deferred entirely: editing/deleting scans, non-E4L scan sources.

## Components

### 1. Prod store — `client_scans` table (`LOG_DB`)

New module `dashboard/client_scans.py`:
```
client_scans (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  email      TEXT NOT NULL,     -- lowercased
  scan_date  TEXT NOT NULL,
  scan_id    TEXT,
  synced_at  TEXT,
  UNIQUE(email, scan_date)
)
```
Functions: `init_client_scans_table(cx)`; `upsert_scans(cx, email, scans)` where `scans = [{scan_date, scan_id}]` (idempotent per `(email, scan_date)`); `scans_for(cx, email) -> [{scan_date, scan_id}]` (most-recent first). Emails lowercased/stripped.

### 2. Sync endpoint — `POST /api/console/client-scans/sync` (`_portal_console_ok()`-gated)

Body: `{email, scans: [{scan_date, scan_id}, ...]}` (one client per call, or a `batch: [{email, scans}]` for efficiency). Upserts into `client_scans`. Returns `{ok, upserted}`. Owner-only; not behind `SCAN_LIST_ENABLED` (the sync populates data regardless of whether the portal surface is on yet).

### 3. Local sync script — `~/AI-Training/02 Skills/e4l-scan-manifest-push.py`

Mirrors the existing `console-push.py` pattern (urllib POST to `glen-knowledge-chat.onrender.com`, `CONSOLE_SECRET` from Doppler `remedy-match/prd`). Reads `~/AI-Training/e4l.db` (read-only): `SELECT c.email, s.scan_date, s.scan_id FROM e4l_scans s JOIN e4l_clients c ON s.client_id=c.client_id WHERE c.email != ''` (confirm the real join column + email field), groups by lowercased email, POSTs each client's manifest (batched) to the sync endpoint. Idempotent — safe to re-run. **Trigger:** piggyback the existing E4L ingest cron (`e4l-daily-watch.sh` / the launchd path that already runs after ingest) so prod stays fresh as new scans land; also runnable manually for the initial backfill.

### 4. Portal listing — `api_client_portal` + `client-portal.html`

- **Payload:** when `SCAN_LIST_ENABLED`, add `available_scans` = for `email_for_reports`, `client_scans.scans_for(email)` annotated with `processed` = the date is in `portal_biofield_reports.list_report_dates(email)`. Shape: `[{scan_date, scan_id, processed}]`, most-recent first. Best-effort — a failure omits the section, never breaks the portal.
- **Render:** a **Scan history** section listing every scan date. A `processed` date reuses the existing report card (openable; read-receipt New/date if that flag is also on). An unprocessed date renders as a dated row labeled "Available — not yet analyzed" (no action in A; B adds the request button). Composes with the household `?member=` switcher — because the list keys on `email_for_reports`, a caregiver viewing a member sees that member's full scan history. Flag-off → no section (portal unchanged).

## Data flow

Ingest cron updates `~/AI-Training/e4l.db` → `e4l-scan-manifest-push.py` reads `e4l_scans⋈e4l_clients` → POST `/api/console/client-scans/sync` → `client_scans` (prod) → `api_client_portal` reads `scans_for(email)` + marks processed via `list_report_dates` → portal renders the Scan history section.

## Files

- **Create** `dashboard/client_scans.py` — table + `upsert_scans`/`scans_for`.
- **Modify** `app.py` — `POST /api/console/client-scans/sync`; `available_scans` in `api_client_portal` payload (behind `SCAN_LIST_ENABLED`); a `_scan_list_enabled()` helper.
- **Modify** `static/client-portal.html` — the Scan history section.
- **Create** `~/AI-Training/02 Skills/e4l-scan-manifest-push.py` — the local sync (+ wire into the ingest cron trigger).
- **Test** `tests/test_client_scans.py`.

## Error handling

- `SCAN_LIST_ENABLED` off → no `available_scans` key, no section, portal byte-identical.
- Sync endpoint: `_portal_console_ok()`-gated; malformed body → 400; partial batch failures logged, others upserted.
- Portal payload best-effort — a `client_scans` read failure omits the section.
- Local script read-only on `e4l.db`; missing/locked DB → logs + exits non-zero (like the other E4L scripts); idempotent re-runs.
- A client with several E4L emails: each email's scans list under that email; the portal shows whichever email the token/`?member=` resolves to.

## Testing

- **`client_scans.py`:** `upsert_scans` inserts + is idempotent per `(email, scan_date)`; `scans_for` returns most-recent-first; emails lowercased.
- **Sync endpoint:** upserts a manifest; `_portal_console_ok()` 401 unauth; batch form; re-sync doesn't duplicate.
- **Payload:** `available_scans` present only when `SCAN_LIST_ENABLED`; each item's `processed` matches whether a published report exists for that date; `?member=` yields the member's list; flag-off → key absent.
- **Render (static):** Scan history lists processed (openable) vs available (labeled) rows; escaped; flag-off → no section.

## Out of scope (→ B or later)

- The "request further analysis" action, the 1/mo-free / unlimited-paid rate gate, and fulfillment (synthesis + publish) — sub-project B.
- Real-time sync (v1 is cron-cadence); editing scans; non-E4L sources.
