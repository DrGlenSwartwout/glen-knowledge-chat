# People Hub Phase 3 (increment 1): graduate personal-email engine to opted-in clients

**Date:** 2026-06-13 · **Status:** approved, implementing

## Context

Phases 1–2 built the segmented, consented People hub and mirrored opted-in
contacts to GHL. Phase 3 = automated communications per segment. Decision (with
Glen): **graduate the existing AI personal-email engine** (`incentive_engine.py`)
from its 4-person hardcoded beta cohort to the real People-hub segments, starting
with **opted-in clients** (retention/reorder), ramped for deliverability.

The engine already does the hard part: Claude-personalized body+subject,
engagement-gated cadence (`should_send_today`), anti-stale topic selection, SMTP
send, send/state tracking. It just needs segment-aware recipient selection, a
volume ramp, and a working unsubscribe.

## Changes

### incentive_engine.py
- **`_list_segment_cohort(segment_tags, audience, cap)`** — query `people` for `type:client` AND `consent:opted-in` (JSON-quoted exact match), skip any suppression tag (defense-in-depth), require an email. Bootstrap each into `users` (auth_method `segment-bootstrap`) and `personal_email_state` (set `audience_tag`). Mirrors `_list_beta_cohort_users`.
- **`_process_one_user(user, state, config, audience, dry_run)`** — extract the per-user generate→send→record body from `run_daily_send_for_beta_cohort` so both orchestrators share it. `dry_run` generates but doesn't send/record.
- **`run_daily_send_for_segment(segment_tags, audience, cap, dry_run)`** — iterate the cohort, apply the engagement gate, send via `_process_one_user`, **stop at `cap` sends** (the ramp). Returns `{sent, skipped, capped}`.
- **Unsubscribe primitives:** `unsubscribe_token(email)` / `verify_unsub_token` (HMAC over a secret) and **`revoke_consent(email)`** — remove `consent:opted-in`, add `consent:unsubscribed` in `people.tags`, and pause the user's `personal_email_state`.

### generate_personal_email
- Point `unsubscribe_url` at `{PUBLIC_BASE_URL}/unsubscribe?email=…&channel=personal&t={token}` (was a hardcoded onrender URL, no token).

### templates/personal_email.txt.j2
- Label the footer link clearly: `Unsubscribe: {{ unsubscribe_url }}`.

### app.py
- **`/cron/personal-send`** — if env `PERSONAL_EMAIL_SEGMENT` is set (e.g. `type:client`), call `run_daily_send_for_segment(... cap=SEND_CAP ...)`; else keep the beta cohort (default, unchanged). Honor a `dry_run` param.
- **`GET /unsubscribe`** — params `email`, `channel`, optional `t`; verify token if present, `revoke_consent(email)`, return a simple confirmation page. Idempotent.

## Safety / rollout
- **Feature-flagged:** beta cohort stays the default until `PERSONAL_EMAIL_SEGMENT` is set. Segment sends are opt-in via env.
- **Ramp:** `SEND_CAP` (default small, e.g. 25/run) caps sends per run so a 2,000-person segment is introduced gradually, not blasted.
- **Dry-run** path for verification before any live send.
- **Prerequisite (ops):** `SMTP_HOST/USER/PASS` and `PUBLIC_BASE_URL` set in env, else `_send_email` logs to console (no real send). Verify before flipping the flag.

## Out of scope (later increments)
Prospects track, newsletter channel, product-affinity selection, GHL workflows,
token-hardening beyond basic HMAC, per-segment cadence tuning.

## Tests (`tests/test_phase3_client_emails.py`)
- `_list_segment_cohort`: returns only opted-in clients (excludes cold client, opted-in prospect, suppressed), bootstraps users + `audience_tag='client'`.
- Cap: orchestrator sends ≤ cap even with more eligible; `_process_one_user` honors the engagement gate; `dry_run` sends nothing.
- Unsubscribe: token round-trips; `revoke_consent` removes opted-in, adds unsubscribed, pauses; `/unsubscribe` endpoint revokes and is idempotent.

## Verification
`/cron/personal-send?dry_run=1` with `PERSONAL_EMAIL_SEGMENT=type:client` → counts (cohort size, would-send under cap), no sends. Confirm SMTP/PUBLIC_BASE_URL set. Then a tiny live cap, inspect `personal_email_sends`. Click an unsubscribe link → confirm `consent:opted-in` removed + `consent:unsubscribed` added + user paused.
