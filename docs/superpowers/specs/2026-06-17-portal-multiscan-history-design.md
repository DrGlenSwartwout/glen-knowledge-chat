# Spec: Portal multi-scan biofield history + date navigation

**Date:** 2026-06-17
**Status:** Approved (design) — pending implementation plan
**Related:** [[project_ascension_pricing_model]] · [[project_e4l_scan_ingestion]] · [[project_unified_personal_portal]] · builds directly on the E4L auto-draft + blur-reveal feature (PR #156, merged) — the single-report state machine this generalizes to per-scan.

---

## Goal

A client's portal keeps **every** biofield scan as its own dated report, navigable by date tabs (newest default). Each scan carries its own blur-reveal status; recent scans are actionable, older ones are read-only history. The client can return to any past scan and see that report "in its most complete form for that date." Editing/confirming a draft is captured as a correction for AI training, which is why the in-portal copy of the raw auto-draft is not separately retained.

## Decomposition — this is one cohesive subsystem

In scope: the `portal_biofield_reports` store, scan-date-aware read/transition/editor endpoints, the 30-day actionability window, the date-tab UI, and the training-correction capture. Deferred (unchanged from PR #156's deferrals): access-gating/quotas, the $99 offer/checkout, purchase state. Those layer on top; this spec leaves their hooks intact (per-scan GHL tags still fire).

## Data model

New table, additive — the biofield analysis becomes per-scan:

```
portal_biofield_reports
  email        TEXT      -- the client (normalized lower/stripped)
  scan_date    TEXT      -- from e4l.db; the tab label + key
  scan_id      TEXT      -- provenance (e4l scan_id)
  content_json TEXT      -- {greeting, video, layers, pricing_note, reorder_items} for THIS scan
  status       TEXT      -- ai_draft | interested | requested | confirmed (per scan)
  created_at, updated_at TEXT
  UNIQUE(email, scan_date)        -- one evolving report per scan date
  INDEX(email)
```

- One **evolving** row per (email, scan_date): it starts as the AI auto-draft and edits overwrite `content_json` in place. The unedited auto-draft is **not** kept here — it is already logged locally by the importer, and the edited version is captured for training (below), so the auto→edited delta is preserved outside the portal.
- `client_portals` is unchanged: it remains the identity/token anchor. Its legacy `content_json` biofield fields stay as the **fallback** (below).
- `content_json` uses the SAME biofield content schema as today (`{greeting, video, layers:[{n,title,meaning,remedy,dosing}], pricing_note, reorder_items}`), just scoped to one scan. `reorder_items` rides with the report so "Order my remedies" reflects the selected scan.

### Back-compat (no migration)
If a client has **zero** rows in `portal_biofield_reports`, the biofield block falls back to `client_portals.content_json` and renders it as a single **confirmed** report (matching today's behavior — legacy/hand-built/test portals are unaffected). In that case `scan_dates` is empty and the page shows the single report **with no date tabs**. Once the client's **first** dated report exists, the reports table is authoritative and the legacy `content_json` is superseded (no longer shown) — a new scan supersedes any prior hand-built analysis. New auto-drafts and editor publishes write to the reports table; history accrues going forward. No backfill required.

## Components

### 1. Accessors — `dashboard/portal_biofield_reports.py`
`init_table(cx)`, `upsert_report(cx, email, scan_date, scan_id, content, status)`, `get_report(cx, email, scan_date)`, `list_report_dates(cx, email)` (newest-first list of scan_dates), `latest_report(cx, email)`, `set_report_status(cx, email, scan_date, status)`. Self-contained, `cx`-based, mirrors `client_portal.py` conventions.

### 2. Scan-date-aware reads — `app.py`
`GET /api/portal/<token>` and `GET /api/portal/<token>/view` accept optional `?scan_date=` (default = newest report). Response adds **`scan_dates: [<newest…oldest>]`** and the selected report's `scan_date`. The biofield block is built from the selected report (or the legacy fallback when no rows). Server-side blur is enforced per the selected report's status + the actionability rule.

### 3. Actionability window
A report is **actionable** iff `scan_date` is within **30 days of today** (date math on the calendar dates; today supplied by the server).
- Actionable & not `confirmed` → patterns shown, remedies blurred, **CTA** ("View your scan analysis" / "Request my remedy matches") per the existing state machine.
- Older than 30 days → **read-only**: `confirmed` shows full remedies; any non-confirmed status shows blurred patterns + the "AI-generated, pending Dr. Glen's review" line but **no CTA**. The block carries `actionable: false`.

### 4. Transitions carry scan_date — `app.py`
`POST /api/portal/<token>/biofield/interest` and `/request` accept the target `scan_date` (JSON body or query; default newest). They (a) reject with 409 if that report isn't actionable (out-of-window) or doesn't exist, (b) set that report's status, (c) enqueue the GHL tag (`e4l:interested` / `e4l:requested`). Unconfirmed remedies are never returned.

### 5. Editor + review queue — `app.py` + `static/console-biofield-portal.html`
- The console editor loads a specific `scan_date` (default newest) and publishes to that report; **publish confirms that report** (`status=confirmed`) + enqueues `e4l:confirmed` + appends the training correction (below).
- `GET /api/console/biofield/review-queue` returns requested reports as `{email, name, scan_date, requested_at}`, newest-first.

### 6. Auto-draft importer — `02 Skills/e4l-portal-import.py` + `/admin/portal/upsert`
`--publish-draft` POSTs the content **with `scan_date` + `scan_id`** (from `e4l.db`) and `biofield_status: ai_draft`. `/admin/portal/upsert` is extended: when `scan_date` is present it ensures the `client_portals` token row exists AND upserts a `portal_biofield_reports` row (status from `content.biofield_status`); without `scan_date` it behaves exactly as today (legacy single-content write). The email-trigger hook (dark behind `E4L_AUTODRAFT_ENABLED`) is unchanged except the importer now sends the date.

### 7. Training-correction capture — `app.py` (+ local pull)
On publish/confirm, append the confirmed `content_json` to a server-side `biofield_corrections` log (a small table: `email, scan_date, content_json, created_at`). A console endpoint `GET /api/console/biofield/corrections?since=<iso>` returns new corrections so the local training pipeline records the auto→confirmed delta against the auto-draft it already logged. This is the mechanism that makes dropping the in-portal auto-draft safe.

### 8. Page UI — `static/client-portal.html`
Date tabs above the biofield section: the latest 3–4 scan_dates as tabs + a **"More dates ▾"** dropdown listing all older dates; newest selected by default. Selecting a date re-fetches `?scan_date=` and re-renders the biofield block. CTAs render only when the selected report has `actionable: true`. Brand rules unchanged (no emojis, "Order" not "Reorder", dark-green/gold, AI-disclosure until confirmed).

## Data flow
scan ingests → importer auto-drafts a **dated report** (ai_draft) → client opens portal, newest tab shown → click 1/2 advance *that scan's* status (within window) → Glen publishes that scan in the editor (confirmed + correction logged) → remedies reveal for that tab. Client can tab to any past scan; recent ones actionable, older read-only.

## Error handling
- Unconfirmed remedies never sent (server omits remedy/dosing) on BOTH endpoints, per selected report.
- Out-of-window transition → 409, no state change.
- Unknown `scan_date` → fall back to newest (reads) / 409 (transitions).
- No reports for the email → legacy `client_portals.content_json` fallback, rendered confirmed (no regression).
- GHL enqueue / correction-log failure → best-effort, logged, never breaks the transition or publish.

## Testing
- **Accessors:** upsert/get/list-dates (ordering)/latest/set-status; UNIQUE(email,scan_date) overwrite.
- **Reads:** `/view` + content endpoint return `scan_dates` newest-first; `?scan_date=` selects the right report; blur per that report's status; back-compat fallback when no rows.
- **Window:** a >30-day non-confirmed report renders blurred + `actionable:false` + no CTA; transitions on it → 409; a recent one is actionable.
- **Transitions:** set the correct report's status + enqueue the dated tag; idempotent.
- **Editor/queue:** publish confirms the targeted scan_date + logs a correction; review-queue lists requested reports with scan_date.
- **Importer:** `--publish-draft` writes a dated report row (mock the POST).
- Deploy-chat pytest conventions + isolation; full suite green.

## Definition of done
Every scan is its own dated report; the portal shows date tabs (newest default, recent tabs + dropdown); each scan blurs/reveals on its own status; scans within 30 days are actionable, older ones read-only; editing/confirming captures the correction for AI training; legacy portals render unchanged. Othon's 6/5 scan is published as the first real report with the invented FF names corrected. Full suite green.
