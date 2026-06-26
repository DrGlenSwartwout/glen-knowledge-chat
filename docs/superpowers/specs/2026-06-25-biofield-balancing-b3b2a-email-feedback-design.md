# Biofield Intake — Balancing Loop B3b-2a: Email-Feedback Mining

**Date:** 2026-06-25
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent:** SP-B live balancing loop. Extends B3b (recent-communication mining, #304). First of the deferred connector sub-increments (B3b-2). B3b-2b Gmail / B3b-2c Practice Better / B3b-2d GHL remain parked.

## Problem

B3b mines a client's chat, inquiries, and ScoreApp intake into stresses. But the practice already has another live, high-signal communication source: **`personal_email_feedback`** — client replies to Glen's emails, ingested by `reply_watcher.py` and **already AI-distilled** into `ai_summary`, `extracted_topics`, and `extracted_conditions`. That content (especially the conditions) is essentially pre-extracted stress material, and today it never reaches the intake stress list.

## Goal

Fold a client's recent email feedback into the existing recent-comms mining so it flows through the same pipeline (`recent_comms` → endpoint → `comms_to_text` → `interpret_stresses` → `add_stress(source='comm')`), with no new endpoint, route, or UI.

## Non-goals

- New connectors for Gmail-per-client (B3b-2b), Practice Better (B3b-2c), or GHL (B3b-2d).
- Mining the raw reply text (`raw_text`) — noisy and higher-PII; we use the distilled fields only.
- Any change to the prod endpoint, local route, button, or `add_stress`/merge behavior — B3b-2a only extends the two pure aggregation/flatten functions.

## Design

### Extend `recent_comms` (dashboard/recent_comms.py)

`personal_email_feedback` and `users` live in the same `chat_log.db` that `recent_comms(cx, …)` already queries, so no new connection. Add a fourth section, windowed like the others:

- New return key `recent_feedback: [ {"summary": str, "topics": [str], "conditions": [str], "received_at": str} ]`.
- Query (best-effort, wrapped like the existing sections; missing table → empty):
  ```sql
  SELECT pf.ai_summary, pf.extracted_topics, pf.extracted_conditions, pf.received_at
  FROM personal_email_feedback pf JOIN users u ON u.id = pf.user_id
  WHERE lower(u.email) = lower(?) AND pf.received_at > datetime('now', ?)
  ORDER BY pf.received_at DESC
  ```
  (`?` = email, then the `f"-{int(days_window)} days"` window string, same int-guard as B3b.)
- `extracted_topics` / `extracted_conditions` are JSON-list TEXT columns — parse each with `json.loads` inside a try/except → `[]` on bad/empty JSON. `ai_summary` is plain text. **`raw_text` is not selected.**
- The existing three sections (`intake_summary`, `recent_inquiries`, `recent_queries`) are unchanged; `recent_feedback` is additive, so the prod `/api/people/recent-comms` endpoint returns it automatically (it just `jsonify`s the dict).

### Extend `comms_to_text` (dashboard/biofield_comms.py)

Append the feedback to the flattened blob the extractor sees, after the existing sections:
- For each `recent_feedback` item: add `ai_summary` (if non-empty), then the `topics` and `conditions` joined as a comma list (these are the strongest stress signals).
- Empty/missing `recent_feedback` → no change (back-compatible; a context without the key behaves exactly as today).

That's the whole change — the feedback text now reaches `interpret_stresses` and becomes `source='comm'` stresses, merged by normalized label like every other comm source.

### Why no other files change

The prod endpoint (`/api/people/recent-comms`) already returns `jsonify(recent_comms(cx, email))`, so a new dict key flows through with zero endpoint change. The local `fetch_recent_comms` passes the dict opaquely to `comms_to_text`. The route, button, hook, `add_stress`, and merge are untouched.

### Components / files

- `dashboard/recent_comms.py` — add the `recent_feedback` section + its query/JSON-parse.
- `dashboard/biofield_comms.py` — `comms_to_text` includes `recent_feedback`.
- (Prod deploys via the unchanged endpoint picking up the new section — same auto-deploy as B3b.)

### Testing (TDD, offline)

`recent_comms` is connection-based (tmp sqlite); `comms_to_text` is pure.
1. **recent_comms feedback section** — seed `users` + `personal_email_feedback`: a within-window row (with JSON `extracted_topics`/`extracted_conditions` + `ai_summary`) is returned parsed; an out-of-window row is excluded; a row for a different user's email is excluded (join by email); bad/empty JSON in the columns → `[]` (no raise); missing `personal_email_feedback`/`users` table → `recent_feedback: []` (best-effort); `raw_text` never appears in the output.
2. **comms_to_text** — a context with `recent_feedback` includes the summary + topics + conditions in the blob; a context without the key is unchanged (back-compat); the existing B3b `comms_to_text` tests stay green.

## Rollout

The `recent_comms` change deploys to prod via the existing `/api/people/recent-comms` endpoint (additive dict key — read-only, no behavior change to the endpoint or other sections). Local tool needs the updated `comms_to_text` to surface the new section. No feature flag. After merge + Render redeploy, the email-feedback signal joins the comm-stress mining automatically. Post-deploy: curl the endpoint for a client with recent feedback and confirm `recent_feedback` is populated.
