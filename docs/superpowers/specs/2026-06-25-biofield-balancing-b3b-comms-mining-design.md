# Biofield Intake — Balancing Loop B3b: Recent-Communication Mining

**Date:** 2026-06-25
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent:** SP-B live balancing loop. Builds on B1 (#295), B2 (#297), B3a (#300), B4 (#301). This is the LAST SP-B increment — the "mine recent communications, especially last 7 days, into stresses woven into reports" requirement.

## Problem

B3a mined the consolidated People-hub profile into stresses. Glen also wants the client's **recent communications** — chat questions, inquiry challenge/goal, and the ScoreApp intake quiz, **especially the last 7 days** — itemized as stresses. That data lives in **prod's** `chat_log.db` (`query_log`, `inquiries`, `inbound_leads.raw_json`), written by the live site and unreachable from Glen's Mac except over HTTP. Prod already has `_member_context_for_email(email)` (app.py:7548) that aggregates these for the chatbot — but with no time window and no read endpoint.

## Goal

Expose a client's windowed recent communications via a read-only prod endpoint, and in the local intake tool mine that text into stresses (`source='comm'`, merged by normalized label), so the balancing loop reflects what the client has actually been saying.

## Non-goals (deferred / out of scope)

- Email feedback (`personal_email_feedback`), GHL notes/messages, Practice Better messages, per-client Gmail — these need connectors that don't exist (or unpopulated tables). A possible later **B3b-2**.
- Changing the chatbot's `_member_context_for_email` (left untouched — no chatbot regression).
- B4 set-cover (done) / B1–B3a behavior.

## Key difference from B1–B4

B3b is the **first SP-B increment that changes prod** (`app.py`): it adds one read-only, console-gated GET endpoint that merges to main and auto-deploys to Render. It is additive and read-only — no existing prod behavior changes. Everything else stays in the local tool.

## Design

### Windowed aggregation (pure, connection-based — testable offline)

New `dashboard/recent_comms.py`:

`recent_comms(cx, email, *, days_window=7, query_log_n=20) -> dict`
- Takes an open sqlite connection (`cx`) so it is unit-testable with a tmp DB — no `app` import, no network.
- Returns `{"intake_summary": str, "recent_inquiries": [{"main_challenge","main_goal","created_at"}], "recent_queries": [{"question","ts"}]}` (same shape as `_member_context_for_email`, minus the unused `voice_scan_summary`).
- **Intake:** latest `inbound_leads` row (source in scoreapp/practice-better/concierge), **age-agnostic** — parses `first_name` + `total_score.percent` + first 6 `quiz_questions[].question`/`.answers[].answer` into `intake_summary` (same parsing as the existing aggregator).
- **Inquiries:** `WHERE client_email=? AND created_at > datetime('now','-{days_window} days') ORDER BY created_at DESC` (days_window bound as an int).
- **Recent queries:** `WHERE email=? AND ts > datetime('now','-{days_window} days') ORDER BY id DESC LIMIT query_log_n`, with the existing `question`→`query` column fallback.
- Each query is wrapped best-effort (a missing table or bad row never raises; that section just stays empty).

### Prod read endpoint (thin)

`GET /api/people/recent-comms?q=<email>` in `app.py`, mirroring the `/api/people` auth (lines 16319-16324): if `CONSOLE_SECRET` set, require `X-Console-Key` header or `?key=` == secret, else 401. Body: `recent_comms(sqlite3.connect(LOG_DB), email)` as JSON. Read-only; no writes. (The route wrapper is ~6 lines and is not offline-testable because importing `app` needs prod credentials; its logic lives in the pure `recent_comms`, which IS tested. Covered by the manual verification step.)

### Local fetch + flatten + mine

- `comms_to_text(context) -> str` in new `dashboard/biofield_comms.py` (pure): flattens `intake_summary` + each inquiry's `main_challenge`/`main_goal` + each `recent_queries[].question` into one newline-joined text blob for the extractor. Empty context → `""`.
- `fetch_recent_comms(email) -> dict` in `biofield_local_app.py` (injectable, default `_default_fetch_recent_comms`): HTTP GET to `{PUBLIC_BASE_URL or https://illtowell.com}/api/people/recent-comms?q=…&key=…` with `X-Console-Key`; returns the dict or `{}` on any error (best-effort, never raises; no network call when `CONSOLE_SECRET` unset, same as B3a's `_default_fetch_profile`).
- `_mine_comms(cx, test_id)` in `biofield_local_app.py`: resolve email from `_report_for`; no email → `{"added":0,"error":...}`; else best-effort `fetch_recent_comms(email)` → `comms_to_text` → `interpret_stresses(text, interpret_complete)` → `add_stress(..., source='comm')`; return `{"added": n}`. Wrapped try/except.

### Route + always-on hook

- `POST /author/<test_id>/mine-comms` → `_mine_comms(...)`.
- `_seed_stresses` also calls `_mine_comms` best-effort, **run-once-guarded on `source='comm'`** (only when no comm stress exists yet for the test), alongside the B3a profile mining — so a header-save does scan-seed + profile-mine + comms-mine, each guarded to fetch once. Runs whether or not a scan exists.

### Source / merge / report

- Comm-derived stresses: `source='comm'`, `balance='required'`, stored `code=_norm(label)`, merged by normalized label across all sources (scan/voice/tag/comm) — no duplicates. (Reuses B3a's `add_stress`.)
- They appear in the stress panel + report listing like any stress (B1/B2). Narrative weaving of the raw comm text is **out of scope** here (B3a already weaves the profile; comm text weaving can be a later polish).

### UI

A "Mine recent comms → stresses" button near the stress panel + `mineComms()` JS (POST `/author/__TID__/mine-comms` → `loadStress()`), mirroring B3a's mine-profile button.

### Components / files

- `dashboard/recent_comms.py` (new) — `recent_comms(cx, email, *, days_window=7, query_log_n=20)`.
- `dashboard/biofield_comms.py` (new) — `comms_to_text(context)`.
- `app.py` — `GET /api/people/recent-comms` (thin, calls `recent_comms`).
- `biofield_local_app.py` — `_default_fetch_recent_comms`, injectable `fetch_recent_comms`, `_mine_comms`, `POST /author/<id>/mine-comms`, seed-hook call.
- `dashboard/biofield_report_html.py` — "Mine recent comms → stresses" button + `mineComms()`.

### Testing (TDD, offline)

`recent_comms` and `comms_to_text` are pure (tmp sqlite / dict); `fetch_recent_comms`, `interpret_complete` injected.
1. **recent_comms** — seeds tmp `query_log`/`inquiries`/`inbound_leads`: chat + inquiries older than the window are excluded, within-window included; the latest intake quiz is always included regardless of age; ScoreApp Q&A parsed; `question`/`query` column fallback works; missing table → that section empty (no raise).
2. **comms_to_text** — flattens intake + challenge/goal + queries; empty context → "".
3. **mine-comms route** — stubbed `fetch_recent_comms` + `interpret_complete` → comm stresses added (`source='comm'`) and visible in `/stresses`; no-email → error; empty comms → `{"added":0}`; failure → best-effort; merge dedups against an existing scan/voice/tag stress.
4. **always-on hook** — a header-save with stubbed comms mines once (a `source='comm'` stress appears) and is run-once-guarded.
5. **UI** — button + `mineComms()` render and post to the route.

The `app.py` route wrapper is verified manually (it can't import offline); its logic is covered by the `recent_comms` unit tests.

## Rollout

`app.py` change merges to main and auto-deploys to Render (read-only, console-gated endpoint). The local tool changes are inert until the endpoint is live. No feature flag. After deploy, confirm `/api/people/recent-comms?q=<email>&key=$CONSOLE_SECRET` returns the windowed dict, then the local "Mine recent comms" button + header-save hook light up.
