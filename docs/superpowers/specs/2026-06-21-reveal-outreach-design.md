# Reveal Outreach: Send the Reveal Link to Seeded Clients

**Date:** 2026-06-21
**Repo:** deploy-chat (Flask, illtowell.com).
**Status:** Approved (design); ready for implementation plan.
**Parent:** Completes the backfill loop ([[project_e4l_reveal_push]] sub-project C). The backfill silently seeds reveal drafts (`notify=false`, no email), so an approved seeded draft currently has no way to reach the client. This adds the deliberate outreach step.

---

## Problem

Silently-seeded reveal drafts (15 today, more from sub-project C and any future silent push) never emailed the client. After Glen reviews and approves one, there is no way to send the client their "your analysis is ready" magic link. The drafts are stranded.

## Goal

Give Glen a deliberate way to email a reviewed client their reveal link: a per-draft "Send reveal link" button and a "Send all approved un-notified" batch, in the console. Approve stays send-free; outreach is a separate, intentional action; each client is emailed at most once unless Glen re-sends.

## Scope

- A `notified_at` column on `biofield_reveals` + `set_notified` + `list_approved_unnotified`.
- A shared `_send_reveal_link(cx, rid)` (mint fresh token, set on the reveal + `auth_tokens`, send the "ready" email, mark notified) - factored out of the existing `_resend_biofield_reveal` so both reuse it.
- Two console actions: `biofield_reveal.send` (one, approved-gated) and `biofield_reveal.send_all` (batch over approved + un-notified).
- The ingest pre-marks `notified_at` when it actually sends (`is_new and notify`), so matcher/B drafts never appear as un-notified.
- Console UI: per-card Send button + a Send-all button.

**Out of scope:** any change to approve/blur logic, the $1/cart/cancel flow, the matcher, or sub-projects A/B/C. SMS. Scheduling/drip (one-shot send only).

---

## Confirmed decisions (Glen, 2026-06-21)

- **Both:** a per-draft Send button AND a batch "Send all approved un-notified."
- **Approve stays send-free** - outreach is a separate deliberate action (the reason silent seed was chosen: control).
- **Approved-only outreach** - an un-approved reveal still has blurred remedies; only send a reviewed one. (Resend, a different path, is not approval-gated.)
- At-most-once by default (`notified_at`); a manual re-send is allowed per draft.
- No emoji, no em dashes.

---

## Architecture

### 1. Store - `dashboard/biofield_reveals.py`
- Add `notified_at TEXT` via an idempotent additive ALTER (same pattern as `dropped`/`layers_json`). `_row` passes it through (plain column).
- `set_notified(cx, rid)` -> set `notified_at = now`.
- `list_approved_unnotified(cx, limit=200)` -> `SELECT * FROM biofield_reveals WHERE first_approved=1 AND (notified_at IS NULL OR notified_at='') ORDER BY id DESC`.

### 2. Shared send - `_send_reveal_link(cx, rid) -> bool` (app.py)
Factor the reveal mint-and-send out of `_resend_biofield_reveal` (app.py:334):
- Load the reveal by `rid`; if missing -> return False.
- Mint a fresh token; `set_token(cx, rid, hash)`; insert `auth_tokens (biofield_reveal, 30 d)`.
- Build `url = {PUBLIC_BASE_URL}/begin/biofield/{tok}` + the "Your Biofield Analysis is ready" body; `sent = _send_inquiry_email(email, "Your Biofield Analysis is ready", body)`.
- If `sent` -> `set_notified(cx, rid)`; return `bool(sent)`. (Mark only on a successful send, so a failed send stays un-notified and retryable.)
- **No approval gate here** - the caller decides (outreach is approved-only; resend is not).
- Refactor `_resend_biofield_reveal(email, extra)` to resolve the latest reveal id for `email` and call `_send_reveal_link(cx, rid)` (behavior preserved; now DRY).

### 3. Console actions - `dashboard/biofield_reveal_actions.py`
- `biofield_reveal.send {id}`: load the reveal; if not `first_approved` -> `{"sent": False, "reason": "not_approved"}`; else `{"sent": _send_reveal_link(cx, rid)}`. (Inject `_send_reveal_link` via the existing `configure(**deps)` hook so the action module stays import-light - or call it through a passed dependency, mirroring how the module is wired today.)
- `biofield_reveal.send_all`: `rows = list_approved_unnotified(cx, limit=50)`; for each, `_send_reveal_link` wrapped in try/except (log + continue); return `{"sent": n_sent, "of": len(rows)}`. Cap 50/call (well under the 120 s worker timeout for the ~15 case).
- Register both alongside `biofield_reveal.approve` / `.delete` (OWNER/OPS, LOW_WRITE), same `register()` guard.

### 4. Ingest pre-mark - `api_e4l_reveal_draft` (app.py ~11024)
In the `if is_new and notify:` block, after a successful `_send_inquiry_email`, `set_notified` the reveal (reopen a short `LOG_DB` connection there, since the send runs after the main `with` block). So `notify=true` drafts (matcher / sub-project B) are marked notified and never enter the un-notified batch. `notify=false` (backfill) leaves it null.

### 5. Console UI - `static/console-biofield-reveals.html`
- On each **approved** card (the approved section): a "Send reveal link" button. If `d.notified_at` -> label "Sent <date> (re-send)"; else "Send reveal link". `doSend(id)` -> POST `biofield_reveal.send` -> reload.
- A top-of-page **"Send all approved un-notified"** button -> `doSendAll()` -> POST `biofield_reveal.send_all` -> show `{sent} of {of}` -> reload.
- XSS-safe (textContent). The console list payload already includes `notified_at` (via `_row`).

### Reuse / untouched
- `set_token`, `auth_tokens`, `_send_inquiry_email`, `approve_first`, the reveal page, the dispatch spine. Approve/blur/$1/cart unchanged.

---

## Data flow
1. Backfill seeds drafts silently (`notified_at` null).
2. Glen reviews + approves (un-blurs top remedy; sends nothing).
3. Glen clicks Send (one) or Send-all (batch): `_send_reveal_link` mints a fresh token, emails the client the reveal link, marks `notified_at`.
4. The client opens the link -> the reveal (now approved -> top remedy visible) -> the funnel ($1 / cart).
5. matcher/B drafts (notify=true) were already marked at ingest, so they never appear in the un-notified set.

## Error handling
- Send to an unapproved draft -> `{"sent": False}`, no email (approve first).
- A failed `_send_inquiry_email` -> `notified_at` stays null -> retryable; the batch picks it up next run.
- `send_all` isolates per-client errors and caps at 50; returns the count actually sent.
- Re-clicking Send re-sends deliberately and updates `notified_at`; the batch never double-sends (filters null).

## Testing
`tests/test_biofield_reveal_send.py`:
- `_send_reveal_link`: an approved reveal -> mints a token, calls the sender, sets `notified_at`, returns True; a failing sender -> `notified_at` stays null, returns False.
- `biofield_reveal.send`: approved -> sent + marked; unapproved -> `{"sent": False}`, no send.
- `biofield_reveal.send_all`: sends to all approved + un-notified, marks them, skips already-notified and unapproved; returns the count.
- **Ingest regression:** `notify=true` push sets `notified_at`; `notify=false` leaves it null.
- **Resend still works** (`test_link_resend`): `_resend_biofield_reveal` after the refactor still mints + sends for an existing reveal.
- Console serve: the Send + Send-all markers present.
- Senders mocked; tmp `LOG_DB`; no emoji; no em dashes.

Run: `doppler run -p remedy-match -c prd -- env DATA_DIR=$(mktemp -d) ~/.venvs/deploy-chat311/bin/python -m pytest tests/test_biofield_reveal_send.py tests/test_link_resend.py tests/test_biofield_layers.py -v`

## Notes
- Ships live on merge (console-gated; no public flag). The only client-facing effect is an email Glen deliberately triggers.
- After merge: in the console, Send-all the 15 seeded drafts you have approved (or Send them individually) to start the outreach.
- A future drip/scheduled follow-up is out of scope (one-shot send).
