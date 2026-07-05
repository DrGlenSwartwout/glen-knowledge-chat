# Request Further Analysis (Sub-project B) — Design

**Date:** 2026-07-04
**Status:** Approved (brainstormed with Glen 2026-07-04)
**Repo:** deploy-chat (+ a local worker in `~/AI-Training/02 Skills/`)

## Summary

Sub-project A shipped the read-only Scan history (every client sees their E4L scan dates as *Analyzed* or *Available*). **B** lets a client turn an *Available* scan into a full report: a **"Request analysis"** action, rate-gated (free members 1/month, paid unlimited), fulfilled **automatically** by a local worker that runs the existing synthesis pipeline and publishes the report. Plus a **proactive new-scan email** — when a new scan lands, the owner (and, via household cc, their caregiver) is invited to analyze it with a one-click link.

## Reuses existing machinery

- **Rate gate:** `dashboard/analysis_quota.py` — `try_claim`/`release`/`claimed_this_month`, one claim per (email, calendar-month), paid bypass via `_is_paid_member`. Already wired for another flow (app.py:16206). B reuses it as-is.
- **Fulfillment:** `dashboard/biofield_reveal_import.synthesize_reveal_layers(email, scan_id, ...)` + `biofield_portal_publish.publish_to_portal(...)` — the exact synthesis+publish the local `e4l-reveal-push.py` already runs per scan.
- **Scan source:** A's `client_scans` (synced dates) + `available_scans` payload; the household `?member=` switcher; the sharing cc-routing (`household.cc_recipients_for`, #595).

**Not touched:** the existing published-report `status="requested"` flow (`_biofield_transition`, app.py:16175) — that operates on published, actionable-30-day reports with manual console fulfillment; B is a *separate* queue for *unpublished* scans with *automated* fulfillment. They coexist and both reuse `analysis_quota`.

## Decisions (settled with Glen)

- **Quota keyed on the scan OWNER** (`email_for_reports`) — a caregiver requesting a member's scan spends the *member's* monthly slot.
- **Automated fulfillment**, piggybacked on the existing **5-minute** E4L ingest trigger (`e4l-email-trigger`) — a request appears in the portal within ~5 minutes.
- **Proactive new-scan email**, anti-nag gated: only sent when the owner **can act** (paid, or a free member with their monthly slot unused), at most once per scan.

## Scope

**v1 = the request mechanism + portal button + automated worker + proactive new-scan email.** Behind `SCAN_REQUEST_ENABLED` (default OFF). Deferred: re-analysis of already-published reports (the existing flow covers that), request history/analytics, per-scan pricing beyond the tier gate.

## Components

### 1. Request queue — `analysis_requests` (`LOG_DB`)

New `dashboard/analysis_requests.py`:
```
analysis_requests (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  email        TEXT NOT NULL,     -- scan owner (lowercased)
  scan_id      TEXT,
  scan_date    TEXT NOT NULL,
  requested_at TEXT,
  status       TEXT NOT NULL,     -- 'pending' | 'done' | 'failed'
  fulfilled_at TEXT,
  UNIQUE(email, scan_date)
)
```
`init_...`; `create_request(cx, email, scan_id, scan_date) -> {"created":bool,"status":str}` (idempotent per (email, scan_date) — a re-request of a pending/done scan is a no-op returning the current status); `pending(cx, limit=50)`; `mark(cx, id, status)`; `has_pending(cx, email, scan_date) -> bool`.

### 2. Request endpoint — `POST /api/portal/<token>/request-analysis {scan_id, scan_date}`

Behind `SCAN_REQUEST_ENABLED`. Resolves `email_for_reports` with the same `?member=` household authorization as `api_client_portal` (a caregiver may request for a linked member). Then:
1. If the scan is already processed (a published report exists for that date) or already has a pending/done request → return `{ok:true, status:"already"}` (idempotent, no quota spent).
2. **Quota gate (scan owner):** if `_is_paid_member(email_for_reports)` → bypass; else `analysis_quota.try_claim(cx, email_for_reports)` — on failure return `{ok:false, reason:"monthly_quota", upgrade_url:...}` (HTTP 200; the UI shows the upgrade path).
3. On success, `analysis_requests.create_request(...)` (pending). Return `{ok:true, status:"pending"}`.
Token-scoped; a caller can only request for their own or an authorized member's scan.

### 3. One-click email action — `GET /portal/<token>/analyze?scan_id=&scan_date=`

A no-login landing page (token-scoped) that runs the same request logic and renders a small result: "Your analysis is being prepared — it'll appear in your portal shortly" (queued), or an upgrade prompt (quota exhausted / already used), or "already analyzed." GET is acceptable here because the action is idempotent and quota-guarded (a link-prefetch that claims a slot is prevented by requiring the page's own confirm click to POST — the GET lands the page; a button on it POSTs `/request-analysis`). This avoids an email-scanner prefetch silently consuming the monthly slot.

### 4. Portal — Scan history request affordance

A's `available_scans` gains a `requested` flag (a pending `analysis_requests` row exists for that date). Row states in the Scan history section: **Analyzed** / **Requested** (pending) / **Available** + a **"Request analysis"** button. The button POSTs `/request-analysis` (data-attr + addEventListener, per the file's convention); on `pending` the row flips to "Requested"; on `monthly_quota` an inline "1 free analysis per month — upgrade for unlimited" with the upgrade link. Behind `SCAN_REQUEST_ENABLED`.

### 5. Proactive new-scan email (server-side, on sync)

A's `POST /api/console/client-scans/sync` already upserts manifests. Extend it: for a **newly-inserted** scan row (a genuinely new scan, not an update), fire a best-effort "new scan — want it analyzed?" email to the scan owner, **once** (tracked via a new `client_scans.notified_at` column), **only if the owner can act** — `_is_paid_member(email)` OR `not analysis_quota.claimed_this_month(email)`. The email states their limit + the upgrade link and carries the one-click `/portal/<token>/analyze?scan_id=...` link. It is cc'd to caregivers via `household.cc_recipients_for(email)` (a private separate copy, per #595) — so a caregiver is prompted about a dependent's new scan even when the dependent's own email is inactive. Respects `email_suppression` + `notify_state` opt-out. Behind `SCAN_REQUEST_ENABLED`.

### 6. Worker console endpoints (owner-gated, `_portal_console_ok()`)

- `GET /api/console/analysis-requests?status=pending` → the pending queue (`[{id, email, scan_id, scan_date}]`).
- `POST /api/console/analysis-requests/<id>/complete {status}` → mark `done` (worker calls after publishing) or `failed`.

### 7. Local worker — `~/AI-Training/02 Skills/e4l-analysis-fulfill.py` (vault)

Polls `GET /api/console/analysis-requests?status=pending`; for each, runs `biofield_reveal_import.synthesize_reveal_layers(email, scan_id, ...)` against the local `e4l.db` and `publish_to_portal(...)` (reusing the `e4l-reveal-push` pipeline); on success POSTs `/complete {status:"done"}` (the published report makes A's `processed` flip to true → the portal shows "Analyzed"); on synthesis failure POSTs `failed` and logs. **Trigger:** piggyback `e4l-email-trigger.sh` (the 5-minute ingest) so requests fulfill within ~5 min. Idempotent; a `done` request is skipped.

## Data flow

Client (or caregiver via `?member=`) clicks Request analysis / the email link → `POST /request-analysis` → quota gate (scan owner) → `analysis_requests` pending. Separately, a new scan syncs → sync endpoint emails the owner (cc caregivers) with the one-click link. The 5-min worker polls pending → synthesize + publish → mark done → the report appears in the portal (A flips it to Analyzed).

## Files

- **Create** `dashboard/analysis_requests.py`; **Test** `tests/test_analysis_requests.py`.
- **Modify** `app.py` — `_scan_request_enabled()`; `POST /api/portal/<token>/request-analysis`; `GET /portal/<token>/analyze`; `available_scans` gains `requested`; the sync endpoint fires the new-scan email; `GET/POST /api/console/analysis-requests...`.
- **Modify** `dashboard/client_scans.py` — `notified_at` column + a helper to fetch/flag newly-inserted rows + `mark_notified`.
- **Modify** `static/client-portal.html` — the Request button + states + quota message.
- **Create** `static/portal-analyze.html` (or a small inline page) — the one-click result page.
- **Create** `~/AI-Training/02 Skills/e4l-analysis-fulfill.py` (vault) + wire into `e4l-email-trigger.sh`.

## Error handling

- `SCAN_REQUEST_ENABLED` off → no request button, endpoints inert, no new-scan email; A unchanged.
- Quota claim + request insert are transactional: if the request insert fails after a claim, `analysis_quota.release`.
- Worker: a synthesis failure marks the request `failed` (not stuck pending forever) + logs; the client keeps their consumed slot unless the owner releases it from the console (a stuck/failed request is a console item). Best-effort — a fulfillment error never affects the ingest.
- The new-scan email is best-effort (never blocks the sync); `notified_at` prevents re-emailing; suppression/opt-out respected.
- One-click GET lands a page; the actual claim happens on the page's confirm POST → scanner prefetch can't consume a slot.

## Testing

- **`analysis_requests.py`:** create/idempotent per (email,scan_date); pending/mark/has_pending.
- **Request endpoint:** paid bypasses quota; free claims once then `monthly_quota`; already-processed/already-pending → no quota spent; `?member=` requests for an authorized member (quota on the member) and rejects an unlinked member; flag-off inert.
- **available_scans `requested` flag:** reflects a pending request.
- **Sync new-scan email:** fires on a new insert only, once (notified_at), only when the owner can act (paid or unused slot); cc's caregivers; suppressed/opt-out skipped; flag-off no email.
- **Worker endpoints:** pending list + complete (done/failed); owner-gated.
- **Worker script:** unit-test the poll/fulfill loop with an injected synthesis runner (no real e4l.db/LLM); marks done on success, failed on exception.

## Out of scope / future

- Re-analysis of already-published reports (existing `status="requested"` flow).
- Unifying the two console queues (existing "requested" + new analysis_requests) into one view.
- Request analytics; per-scan pricing; quota-reset overrides.
