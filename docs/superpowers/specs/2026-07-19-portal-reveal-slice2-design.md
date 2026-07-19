# Portal-Reveal Unification — Slice 2: Auto-provision portal + portal link in the reveal email

**Date:** 2026-07-19
**Status:** spec for review
**Part of:** reveal-in-portal unification (Option 2 — keep the funnel; single source of truth = System A; backfill later). Slice 1 (portal renders the reveal) is shipped (#1017); this slice makes reveal clients actually HAVE a portal to see it in.

## Scope

On a new biofield scan, (1) auto-provision a **bare** portal for the client's email — a `client_portals` row with a stable token and **no** `portal_biofield_reports` row — and (2) **augment** the reveal-ready email with a portal link *alongside* the existing funnel reveal link. Going-forward only (backfill is slice 3). Behind a flag so it can be rolled out deliberately.

**Why:** slice 1's portal-reveal only activates for a client who has a reveal, a portal, and no System B report. Today no such client exists because the only ways to provision a portal also write a System B report. This slice creates exactly that state for new scans.

## Decisions (locked in brainstorming)

- **Augment, not replace.** The funnel link (`/begin/biofield/<tok>`) stays the primary CTA (conversion untouched). The portal link is added as a secondary "your portal lives here" line.
- **Bare provision.** Use `client_portal.ensure_token(cx, email, name)` — it creates the `client_portals` row (+ stable raw token) and writes NO report row. Idempotent: returns the same stable token on repeat calls.
- **Going-forward only.** Only new-scan reveal emails; no backfill.

## Design

### Provisioning primitive
`client_portal.ensure_token(cx, email, name="")` (dashboard/client_portal.py) is the bare-provision primitive: creates a pending `client_portals` row + stable token if none exists, returns the raw token; writes no `portal_biofield_reports`. Portal URL = `portal_link(token)` = `{portal_base()}/portal/{token}` (app.py). The portal token is DISTINCT from the reveal token — do not conflate.

### Flag
`PORTAL_LINK_IN_REVEAL_ENABLED` (env/Doppler), default **off**. When off, behavior is byte-identical to today (no provisioning, no portal line). Merging is therefore safe; Glen flips the flag to roll out.

### Shared helper (app.py)
`_ensure_portal_link(cx, email, name) -> str | None`:
- If the flag is off → return `None` (no provisioning, no link).
- Else best-effort: `client_portal.ensure_token(cx, email, name)` → `portal_link(token)`. Any exception → return `None` (never break reveal ingest/email). Provisioning is idempotent, so calling on every reveal email is safe.

### Email body (augment)
A pure builder `_reveal_email_body(reveal_url, portal_url) -> str`:
- Always includes the existing reveal section (unchanged wording).
- Appends a portal line ONLY when `portal_url` is truthy:
  ```
  Aloha,

  Your Biofield Analysis is ready. View your reading here:
  {reveal_url}

  Your personal client portal — where your scans and matches live — is here:
  {portal_url}

  In wellness,
  Dr. Glen and Rae
  ```
  When `portal_url` is None, the body is exactly today's (no portal paragraph).

### Wiring (both reveal-email send sites)
Both sites send the same "Your Biofield Analysis is ready" email and must both provision + link:
1. `_send_reveal_link(rid)` (app.py:~763-801) — resend/approval path.
2. The inline `is_new and notify` block in `api_e4l_reveal_draft` (app.py:~24950-24968) — ingest path.

In each: inside the existing `_db_lock` DB block (which already has `cx` and the email), after the suppression guard, call `_ensure_portal_link(cx, email, name)` and build the body via `_reveal_email_body(reveal_url, portal_url)`. `name` is best-effort from the existing people/reveal context, else `""`.

## Data
No schema change. `client_portals` already exists; `ensure_token` handles creation. `people` row is created lazily on first portal load by `portal_identity`.

## Constraints / guarantees
- **No System B report** is ever written by this slice (only `ensure_token`, never a publish/`upsert_report`).
- **Idempotent**: re-sending a reveal email (resend action, re-approval) returns the same portal link; never mints duplicate portals.
- **Best-effort**: a provisioning failure yields `portal_url=None` → the email still sends with just the funnel link. Reveal ingest never breaks.
- **Suppression**: provisioning + portal line only happen for non-suppressed emails (the send sites already guard suppression before building the body).
- **Flag-off = no behavior change.**

## Testing
- `client_portal.ensure_token` (characterization, in-memory): creates a `client_portals` row, returns a token, writes NO `portal_biofield_reports` row; a second call returns the same token (idempotent).
- `_reveal_email_body`: with `portal_url` → contains both URLs + the portal line; with `portal_url=None` → byte-identical to the current body (no portal paragraph).
- `_ensure_portal_link`: flag off → None (no provisioning); flag on → returns a `/portal/<token>` URL and provisions; exception path → None.
- Compile + a focused check that both send sites build the body via the helper.

## Out of scope (later slices)
- **Slice 3:** backfill — provision + notify existing reveal clients (batched, dry-run, mass-email-cap-aware).
- **Slice 4:** fold + retire System B; consolidate the portal request button onto `requested_at`; redirect the standalone reveal page.
- Rebuilding funnel conversion inside the portal (only relevant if we ever switch to "replace").
