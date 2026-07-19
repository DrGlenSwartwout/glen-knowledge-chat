# Portal-Reveal Unification — Slice 3: Backfill portals for existing reveal clients (provision-only)

**Date:** 2026-07-19
**Status:** spec for review
**Part of:** reveal-in-portal unification. Slices 1 (portal renders reveal) + 2 (auto-provision on new scans, flag-gated) shipped. This slice migrates the EXISTING reveal base into portals.

## Scope

Provision a **bare** client portal for every existing `biofield_reveals` email that does not already have one — **provision-only, NO emails sent** (Glen's decision (a): silent backfill; a mass "here's your portal" announcement is a separate, deliberately-triggered tool, not this slice). Exposed as a **dry-run-default, idempotent** console endpoint so it can be run safely and re-run.

**Why provision-only:** mass-emailing the whole reveal history hits the Gmail ~500/day cap (a backfill blew it before) and risks spam complaints on cold addresses. Provisioning is a pure DB migration with zero outward blast. Backfilled clients get their portal link naturally on their next reveal email (slice 2, once the flag is on) or via the existing resend-portal-link tooling.

## Design

### Core function (testable)
`dashboard/portal_backfill.py::backfill_portals(cx, commit=False, limit=None) -> dict`:
- Read distinct lowercased emails from `biofield_reveals`.
- For each: if a `client_portals` row already exists (`SELECT 1 FROM client_portals WHERE email=?`) → count `already`. Else → count `to_provision`; and if `commit`, call `client_portal.ensure_token(cx, email, "")` (bare provision, no report row) → count `provisioned`.
- `limit` caps how many NEW portals are provisioned this run (for safe incremental runs); emails beyond the limit that lack a portal are left for a later run and counted in `remaining`.
- Returns `{"reveal_emails": N, "already": A, "provisioned": P, "remaining": R, "committed": bool}`. Dry-run (`commit=False`) reports `to_provision` under `provisioned=0` with `remaining` = count that WOULD be provisioned.

### Console endpoint (thin wrapper)
`POST /api/console/portal-backfill` — owner/console gated (same auth as other console endpoints). Query/body: `commit` (default false → dry-run), `limit` (optional int). Opens `LOG_DB` under `_db_lock`, calls `backfill_portals`, returns the stats JSON. Never emails.

## Constraints / guarantees
- **No emails.** This slice sends nothing.
- **No System B report** — provisioning is `ensure_token` only.
- **Idempotent** — re-running skips emails that already have a portal (and `ensure_token` is itself idempotent as a second layer). Safe to run repeatedly.
- **Dry-run default** — `commit` must be explicitly set; a bare call reports counts and changes nothing.
- **Bounded** — `limit` allows provisioning in safe batches.
- Provisioning runs under `_db_lock` (serialized writes).

## Testing
- `backfill_portals` (in-memory sqlite, all three tables): dry-run reports the right `to_provision`/`remaining` and writes nothing; `commit=True` provisions bare portals (client_portals rows created, zero `portal_biofield_reports`); re-run is idempotent (`provisioned=0`, all `already`); `limit` caps provisioning and reports `remaining`.
- Endpoint: compile + (best-effort) an auth + dry-run smoke.

## Out of scope
- Any emailing / announcement (separate opt-in tool).
- Slice 4: fold + retire System B; consolidate the portal request button onto `requested_at`; redirect the standalone reveal page.
