# Console Scan-Pull Button + Email-Independent Recency Poll — Design

**Date:** 2026-07-05
**Status:** Drafted (brainstormed with Glen 2026-07-05, direction "Button + auto-poll" approved)
**Repos:** deploy-chat (console UI + prod endpoints + queue table) and `~/AI-Training/02 Skills/` (Mac-side workers)

## Problem

Sean Luscombe did an E4L scan on 2026-07-05. It never appeared in the Biofield Reveals console. Root cause: the entire ingestion chain is **triggered by E4L emailing Glen a "new scan" notification**. E4L sent no email for Sean's scan, so the Mac cron never scraped it, it never entered `e4l.db`, and no reveal draft was created. (Confirmed: Gmail has no E4L scan email for "Luscombe" in 5 days; the scan *does* exist on E4L — pulling it directly by client name recovered `scan_1037956_2026-07-05.pdf` and pushed reveal draft id=52.)

Two gaps to close:
1. **On-demand recovery** — when Glen knows a client scanned but it hasn't surfaced, he needs a one-click way, from the console, to pull that client's latest scan and turn it into a review draft — without waiting on E4L's email and without touching the terminal.
2. **Systemic safety net** — so the next missed email is caught automatically. Today "6 scans today, only 1 in Reveals" is the norm because drafts are only auto-created for emailed + "engaged" clients.

## What already exists (reuse, do NOT rebuild)

Grounded in prior shipped work — see `2026-07-04-available-scan-list-design.md` (sub-project A), `2026-07-04-request-analysis-design.md` (sub-project B), and the live scripts:

- **`scrape-e4l-http.py`** (`02 Skills/`) — resolves a client by name (`--client-name`, auto-picks the account with the most-recent scan) or id (`--client`), impersonates (`/Clients/Clients/LoginAs/{id}`), lists scans (`/MyScans/List`), downloads only scans **not already in `e4l.db`** (`/Scans/ScanPDF/ScanPDFDownload/{scan_id}`). Bypasses email entirely. This is the recovery tool used to fix Sean.
- **`parse-e4l-scans.py` → `e4l.db`; `bulk-vectorize-e4l-scans.py` → Pinecone** — the parse + vectorize steps.
- **`e4l-reveal-push.py` / `e4l_reveal_lib.py`** — synthesize a scan into reveal layers and `POST /api/e4l/reveal-draft` (creates a PENDING draft in `/console/biofield-reveals`). Supports `--no-notify` (silent draft, no client email). This is the exact call the fix used.
- **`e4l-analysis-fulfill.py`** — the **queue-worker template**. Polls `GET /api/console/analysis-requests?status=pending`, runs `_real_synth(email, scan_id)` (resolve scan in local `e4l.db` → `pull_patterns` → `build_payload`) → `_real_publish` (POST reveal-draft) → `POST .../<id>/complete {status}`. Wired as Step 0 of `e4l-email-trigger.sh`, runs every 5 min under `/tmp/e4l-ingest.lock`.
  - **Its limit (the gap we fill):** `_real_synth` resolves the scan from local `e4l.db`; a scan E4L never emailed was never scraped, so it isn't in `e4l.db`, so this worker fails on it. It needs a **scrape-first** step.
- **`client_scans` prod table + `e4l-scan-manifest-push.py`** — mirrors every `e4l.db` scan-date to prod (portal scan-history). Should be refreshed after any new scrape so sub-project A also reflects newly-pulled scans.

## Design decisions (confirmed / proposed)

1. **Both new mechanisms create silent (`--no-notify`) reveal drafts.** They land in "Needs Review." Notifying the client stays a separate, deliberate step (approve → send, or "Send all approved un-notified"). This decouples *surface-for-review* from *notify-client*.
2. **On-demand pull is targeted** (one client Glen names) — cheap, immediate. **Auto-poll is a bounded sweep** (not a full client walk) — E4L exposes no global recent-scans feed, so a blind poll would impersonate ~884 clients.
3. **Drafts are created for every newly-pulled scan regardless of engagement.** The point is to get all new scans into the review queue. (The existing emailed-path autodraft, which is engagement-gated and can auto-notify, is unchanged.)
4. **Prod endpoints mirror the analysis-requests shapes exactly** — they are the proven template; consistency keeps the worker simple.

---

## Feature 1 — Console "Pull scan" button (on-demand, targeted)

### 1a. UI — `static/console-biofield-reveals.html`

Add a control to the toolbar (built at lines 234-241), inserted **before** the "Send all approved un-notified" button so it renders to its left (flex order = insertion order):

- A text input (`placeholder="client name or email"`) + a `Pull scan` button (`.btn.ghost`).
- On click → `POST /api/console/scan-pull-requests {query}`. Disable + show inline status ("Queued — pulling from E4L, usually a few minutes…").
- Poll `GET /api/console/scan-pull-requests/<id>` every ~10s (cap ~5 min). On `done` → toast (`Pulled — draft ready`) + `loadList()` to surface the new draft. On `failed` → show `message` (e.g. ambiguous name, no scan found). On timeout → "Still working; refresh shortly."

### 1b. Prod store + endpoints — deploy-chat

New module `dashboard/scan_pull_requests.py` + table in `LOG_DB` (mirrors `client_scans`/analysis-requests style):
```
scan_pull_requests (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  query       TEXT NOT NULL,        -- name or email as typed
  status      TEXT NOT NULL,        -- pending | working | done | failed
  requested_by TEXT,                -- console user, best-effort
  scan_id     TEXT,                 -- filled on done
  draft_id    INTEGER,              -- reveal draft id on done
  message     TEXT,                 -- failure reason / note
  created_at  TEXT,
  updated_at  TEXT
)
```
Functions: `init_table(cx)`, `enqueue(cx, query, requested_by) -> id`, `list_pending(cx, limit)`, `mark(cx, id, status, **fields)`, `get(cx, id)`.

Endpoints in `app.py` (all `CONSOLE_SECRET`-gated, matching analysis-requests):
- `POST /api/console/scan-pull-requests` `{query}` → `enqueue` → `{id}`.
- `GET  /api/console/scan-pull-requests?status=pending&limit=N` → worker poll → `{requests:[...]}`.
- `POST /api/console/scan-pull-requests/<id>/complete` `{status, scan_id?, draft_id?, message?}` → `mark`.
- `GET  /api/console/scan-pull-requests/<id>` → console status poll → the row.

Behind `SCAN_PULL_ENABLED` (the UI control + the enqueue endpoint; the worker/complete endpoints may stay open like the analysis-requests worker path).

### 1c. Mac worker — `02 Skills/scan-pull-fulfill.py`

Mirrors `e4l-analysis-fulfill.py`. Per pending request:
1. `mark working`.
2. **Resolve query → client name.** If it looks like an email, look up the name in `e4l.db` (`e4l_clients.email`); else use as name. If the name resolves to multiple E4L accounts and `scrape-e4l-http`'s most-recent-scan auto-pick is not confident (ambiguous), `mark failed` with a candidate list in `message` ("ambiguous — enter the email"). (v1 leans on `pick_client`'s existing most-recent-scan tiebreak; email forces exactness.)
3. **Scrape-first:** run `scrape-e4l-http.py --client-name "<name>"` (subprocess, Doppler prd). This downloads any new scan PDF not yet in `e4l.db`.
4. `parse-e4l-scans.py` (with the same small retry the email-trigger uses) → confirm the scan is in `e4l.db`.
5. `bulk-vectorize-e4l-scans.py --batch-size 30`.
6. **Reveal-push (silent):** reuse the `e4l-analysis-fulfill.py` `_real_synth`/`_real_publish` path but with `notify=False`, resolving the client's **latest** scan in `e4l.db` (handles both a brand-new scan and an already-ingested-but-never-drafted scan). Returns the draft id.
7. Refresh the portal manifest for this client (`e4l-scan-manifest-push.py`, best-effort) so sub-project A shows the new scan date.
8. `mark done {scan_id, draft_id}` or `failed {message}`.

Wire into `e4l-email-trigger.sh` as a new step (alongside Step 0 analysis-fulfill), so it runs every 5 min under the existing lock. Failure cap per request (like the email path's `FAIL_CAP`) so an unresolvable query can't retry forever.

---

## Feature 2 — Email-independent recency poll (bounded sweep)

Catches scans E4L never emailed, with no button-press.

### 2a. Discovery constraint + approach

E4L has **no global recent-scans feed**; last-scan-date is only visible per client after `LoginAs` → `/MyScans/List`. So:

- **Spike (first plan task):** confirm whether `/Clients/Clients` (or an E4L admin report) exposes a last-scan / last-activity column or a recent-activity sort. If yes → the poll pulls only the recent-active **head** cheaply. If no → the bounded active-set walk below.
- **Bounded active-set (fallback / default):** the "at-risk" population is recent scanners. Build the set from `e4l.db`: clients with ≥1 scan in the last `RECENCY_WINDOW_DAYS` (default 45), capped at `RECENCY_MAX_CLIENTS` (default 150, newest-activity first). For each, impersonate + `/MyScans/List`, compare E4L's latest scan_id/date against `e4l.db`; on a delta, scrape the PDF → parse → vectorize → reveal-push (`notify=False`) → manifest refresh.

### 2b. Mac job

- `02 Skills/scan-recency-poll.py` + `scan-recency-poll.sh` + launchd `com.glen.e4l-scan-recency.plist`.
- Cadence: nightly to start (tunable; a few-hourly option once cost is measured). Reuses `/tmp/e4l-ingest.lock` (never collides with the email-trigger). Logs `/tmp/e4l-scan-recency.log`. Emits a run summary; reuses the `e4l-ingest-guard.py` alert style if a pulled scan fails to land.
- Behind `SCAN_RECENCY_POLL_ENABLED`. Cost knobs: `RECENCY_WINDOW_DAYS`, `RECENCY_MAX_CLIENTS`, cadence — start conservative, widen after measuring per-run wall-clock and E4L load.

Result: every genuinely new scan becomes a silent "Needs Review" draft regardless of E4L's email — so "6 scans today → all 6 in Reveals."

---

## Data flow

**On-demand:** console button → `POST /api/console/scan-pull-requests` → `scan_pull_requests` (pending) → `scan-pull-fulfill.py` (5-min cron): scrape E4L → parse → vectorize → reveal-push(no-notify) → `POST .../complete` → console poll sees `done` → `loadList()` shows the draft.

**Auto-poll:** `scan-recency-poll.py` (nightly): active set from `e4l.db` → per client impersonate + list → delta → scrape → parse → vectorize → reveal-push(no-notify) → draft in "Needs Review" + manifest refresh.

Both converge on the same sink: a PENDING no-notify draft in `biofield_reveals`, reviewed via the existing console (approve / edit / send unchanged).

## Files

**Create (deploy-chat):**
- `dashboard/scan_pull_requests.py` — table + enqueue/list/mark/get.
- `app.py` — 4 endpoints + `_scan_pull_enabled()` helper.
- `static/console-biofield-reveals.html` — toolbar input + "Pull scan" button + poll JS.
- `tests/test_scan_pull_requests.py`.

**Create (`~/AI-Training/02 Skills/`):**
- `scan-pull-fulfill.py` — on-demand worker.
- `scan-recency-poll.py` + `scan-recency-poll.sh` + `com.glen.e4l-scan-recency.plist` — auto-poll.

**Modify:**
- `e4l-email-trigger.sh` — add the scan-pull-fulfill step.

## Error handling

- **Ambiguous name** → `failed` with candidate list; Glen re-submits with email. (Multiple "Sean"/"Luscombe" accounts exist — real risk.)
- **No scan found on E4L** → `failed` "no scan on file for <client>".
- **Scrape/parse transient miss** → same retry the email-trigger uses; still missing after retries → `failed`, leave for manual, alert (guard style). Never mark a request `done` unless the draft was actually pushed (mirror the email-trigger's "confirm in `e4l.db` before acknowledging" discipline).
- **Draft already exists** for `(email, scan_date)` → `biofield_reveals` UNIQUE upsert refreshes it; report the existing draft id as `done`.
- **Flags off** → button hidden, enqueue endpoint 404/short-circuits, poll job no-ops. Console byte-identical when `SCAN_PULL_ENABLED` off.
- **Auth** → workers use `CONSOLE_SECRET` (Doppler `remedy-match/prd`), same as existing scripts. `WEB_URL` default `glen-knowledge-chat.onrender.com` (matches `e4l-analysis-fulfill.py`).

## Testing

- **`scan_pull_requests.py`:** enqueue/list_pending/mark/get; status transitions; idempotent completes.
- **Endpoints:** enqueue returns id; pending list shape; complete updates row; `CONSOLE_SECRET` 401 unauth; gated by `SCAN_PULL_ENABLED` where applicable.
- **Worker integration (guard against mock-masked green — hit the real push, not a mock mirroring the worker):** enqueue for a client whose latest scan is already ingested → still pushes/refreshes a no-notify draft; enqueue for a client with a genuinely new E4L scan → scrape path lands it. Verify the draft via `GET /api/console/biofield-reveals`.
- **Recency poll:** seed `e4l.db` missing a known recent E4L scan; run the poll over a 1-client active set; confirm scrape + draft; confirm no draft when `e4l.db` is already current.
- **Post-deploy render-verify:** headless-render `/console/biofield-reveals` → the "Pull scan" control renders left of "Send all"; a pull produces a visible draft.

## Rollout

- Ship dark: `SCAN_PULL_ENABLED`, `SCAN_RECENCY_POLL_ENABLED` (both OFF). Deploy, verify on prod, then flip. Env set on the Render dashboard (render.yaml is not the live source) + the plist env for the Mac jobs.
- Sequence: Feature 1 (button + worker) first — it directly serves the case Glen hit and is the cheaper, self-contained piece. Feature 2 (poll) second, after the spike measures E4L discovery cost.

## Appendix — flaky Gmail-check (related, non-blocking)

The email-trigger's Step-1 Gmail check is an LLM (haiku `claude --print`) subprocess. In the last hour its log shows it sometimes returning prose ("I'm unable to invoke the MCP tool…") instead of calling Gmail — i.e. it can report "no new scans" **without actually searching**, a second silent-miss path. Feature 2's poll makes this non-fatal (misses get swept up). Recommended follow-up (separate small workstream, not in this build): replace the haiku Gmail-check with a deterministic Gmail search so the emailed path stops flaking.

## Non-goals

- No console→Mac live bridge beyond the existing poll-a-queue pattern (the console never reaches `e4l.db` directly).
- No auto-notifying clients from either mechanism — drafts only.
- No editing/deleting scans; no non-E4L scan sources.
- No full unbounded E4L client walk.
