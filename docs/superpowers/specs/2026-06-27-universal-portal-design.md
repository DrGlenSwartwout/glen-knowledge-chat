# Universal Portal — Design Spec (sub-project A)

**Date:** 2026-06-27
**Status:** Approved (design), pending implementation plan
**Author:** Glen + Claude
**Builds on:** the portal concierge (#375, merged) — the no-scan home reuses the "Ask Dr. Glen" chat.

## Context

Per the chat surface model (`project_chat_surface_model`), everyone post-TOS should have a portal home. Today a portal is minted **only** when a biofield report is published (console) — so non-biofield remedy buyers have no portal, and the portal concierge (#375) can't serve them. Sub-project A makes the portal **universal**: every remedy buyer gets a portal, agrees to TOS to enter it, and sees a useful home even with no biofield scan. This unblocks the concierge for everyone and lets the funnel concierge retire later.

Decision (Glen): **capture TOS at portal first-entry** (channel-agnostic) rather than bolting a TOS gate onto every external checkout.

## Current state (verified anchors)

- **Single order hook:** every checkout channel (funnel `/begin/checkout`, reorder, practitioner, portal-reorder, GrooveKart webhook, biofield-trial) writes its order through `upsert_order()` (`dashboard/orders.py:81`). One hook there covers all channels.
- **Portal mint:** `client_portal.upsert_portal(cx, email, name, content)` mints a token on first-create (returns `(raw_token, id)`), no-ops on update (`(None, id)`); keyed by email.
- **Link email:** `_send_full_report_email(to, name, subject, body)` (Gmail→SMTP→log cascade, suppression-aware) already sends `/portal/<token>` links on biofield publish.
- **TOS:** recorded as `journey_state.tos_agreed_at` via `begin_funnel.record_unlock(..., trigger="tos")`; `is_member(session_id, email)` is True iff `tos_agreed_at` is set. The funnel checkout already gates on `is_member`; other channels do not.
- **No-scan render today:** `/api/portal/<token>` falls back to base content when no biofield report exists, but the only empty state is `biofield_status: "pending"` → `client-portal.html` shows a "Preparing your Biofield Analysis…" spinner that posts `process-request` and polls (`client-portal.html:301`). For a buyer with no scan this spins forever.

## A1 — Mint portal + email link on purchase

**Hook:** after `upsert_order()` records an order, if the order has a non-empty `email` and **no portal exists** for that email, mint one and email the link.

- New isolated logic (e.g. `dashboard/portal_provision.py`): `ensure_portal_for_buyer(cx, email, name) -> token_or_None` — calls `upsert_portal(cx, email, name, {"biofield_status": "none"})`; returns the raw token **only if newly minted** (so we email once). Idempotent: a repeat order for an existing portal returns None → no email.
- On a newly-minted token, send the `/portal/<token>` link via `_send_full_report_email` ("Your healing home is ready") in a background thread, suppression-aware, **fail-open**: any mint/email error is caught and logged and must NEVER break order ingestion.
- Wiring: call `ensure_portal_for_buyer` from the order path right after `upsert_order` returns, for every channel (it's one call site if all channels funnel through `_ingest_order`; otherwise the smallest shared point). Skip when `email` is empty.
- Do NOT disturb the existing biofield-publish mint (also `upsert_portal`, email-keyed) — a buyer who later gets a scan keeps the same portal/token (update path).

## A2 — No-scan portal home

Introduce `biofield_status: "none"` as the purchase-minted empty state (distinct from `"pending"`, which means a scan is being processed). `client-portal.html` renders, for `"none"`:
- account snapshot + **order history** (existing blocks),
- the **"Ask Dr. Glen" concierge** section (from #375 — `build_context` already grounds on orders alone),
- a soft **"Get your biofield scan"** invitation card linking to the E4L scan (Truly.VIP/E4L),
- **NO** "Preparing your Biofield Analysis…" spinner, NO `process-request` POST, NO polling.

The `"pending"` path (a real scan in flight) is unchanged. `"confirmed"`/scan states unchanged.

## A3 — Portal first-entry TOS gate

When a portal page loads and its email has **no `tos_agreed_at`**, the page shows a TOS agreement step (terms summary + "I agree" action) **before** the home content; agreeing unlocks it.

- A read signal in the portal payload (`/api/portal/<token>` or `/view`): `tos_agreed: bool` for the portal's email (via `is_member(email=...)`).
- New endpoint `POST /api/portal/<token>/agree-tos`: resolves the token → email, records TOS via `record_unlock(..., trigger="tos")` (+ tos_version), returns `{ok:true}`. Fail-open on the recording side is NOT acceptable here (the agreement must persist) — but a DB error returns an error the UI can retry, without crashing.
- `client-portal.html`: if `tos_agreed` is false, render the gate (and suppress the home sections) until the user agrees; on agree → POST → re-render the home. Portals whose email already agreed never see the gate.
- This applies to **every** portal lacking `tos_agreed_at` — including existing biofield clients who never agreed — so the rule is uniform. (Most biofield clients agreed via the funnel and won't see it.)

## Components (isolated)

- `dashboard/portal_provision.py` (new, mostly pure given a `cx`): `ensure_portal_for_buyer(cx, email, name)`. Unit-testable.
- `app.py`: the order-path call to `ensure_portal_for_buyer` + the welcome-email send (background, fail-open); `tos_agreed` in the portal payload; the `POST /api/portal/<token>/agree-tos` route.
- `static/client-portal.html`: the `"none"` home state (concierge + scan CTA, no spinner) + the TOS gate.

## Data flow

1. Buyer checks out (any channel) → `upsert_order` writes the order.
2. `ensure_portal_for_buyer(email, name)` → mints a portal if none → emails the `/portal/<token>` link (once).
3. Buyer opens the link. Portal payload includes `tos_agreed`.
4. If not agreed → TOS gate → `POST /agree-tos` records it → home renders.
5. No-scan home: account + orders + concierge + "get a scan" CTA.

## Error handling / fail-open

- A1 mint + email: fully wrapped; never breaks order ingestion; one email per portal (suppression + first-mint guard).
- A3 agree-tos: persists or returns a retryable error; never silently drops the agreement.
- No-email orders: skipped (no portal minted).
- The portal page render must never break on a missing field.

## Testing

- **Unit (pure-ish):** `ensure_portal_for_buyer` — mints + returns token on first call, returns None on repeat (idempotent), skips empty email; against a tmp DB.
- **Endpoint (Doppler):** `POST /api/portal/<token>/agree-tos` records `tos_agreed_at` for the email; the portal payload's `tos_agreed` flips true after; bad token → 404.
- **Order→mint integration (Doppler):** ingest an order with a new email → a portal token is minted and the welcome-email path is invoked (mock/inspect the send); a repeat order does not re-mint or re-email.
- **Render-verify (headless, gevent server):** a `"none"`-state portal renders account + orders + concierge + scan-CTA with NO spinner/polling; the TOS gate shows for a no-TOS portal and the home shows after agreeing; zero console errors. (gevent — a 1-worker sync server yields `chrome-error://` on concurrent loads.)

## Out of scope

- **Retiring the funnel concierge** — a trivial later step once universal portals are live + adopted.
- **Enforcing TOS at the non-funnel checkouts** — we capture at portal entry instead (per Glen).
- **Backfilling portals for historical buyers** — possible follow-on (a one-off `ensure_portal_for_buyer` sweep over existing `orders` emails); not in the first cut.
- No new commerce.

## Open items for the plan

- Confirm the single shared call site after `upsert_order` (is `_ingest_order` truly the one funnel for all channels, or are there direct `upsert_order` callers to also hook?).
- Confirm `record_unlock`'s exact signature for recording TOS by email without a session.
- Confirm the portal page's gate insertion point + how it suppresses the home sections until agreement.
