# Retire the Standalone Affiliate Portal (Step 2b-3, Option A)

**Date:** 2026-06-26
**Status:** Approved (design — Glen chose Option A)
**Author:** Glen + Claude
**Parent:** affiliate→personal-portal unification. The personal portal now hosts the full ambassador dashboard (2b-1) + social links (2b-2). 2b-3 retires the standalone `/affiliate/portal` and unifies on the personal-portal login, bundling the affiliate→`people` coverage so no affiliate is locked out.

## Problem

The standalone `/affiliate/portal` (its own page + its own magic-link login + affiliate token) duplicates what the personal portal now provides. We want one front door (the personal portal). But the personal portal authenticates via the `people` table (self-login matches `people` by email), and **7 of 8 approved affiliates are in `people`; 1 (`vicsantos336699@`) is not** — a naive redirect would lock that affiliate out.

## Goal

Redirect the standalone affiliate portal + its login into the personal-portal login, and guarantee every approved affiliate has a `people` row so they can self-login. Keep the public affiliate surfaces.

## Design

### Component 1 — affiliate→`people` coverage (reuse `customers.find_or_create_by_email`)

`dashboard/affiliate_dashboard.py: backfill_affiliate_people(cx) -> int` — for each `affiliate_signups` row with `status='approved'` whose email has no `people` row, call `customers.find_or_create_by_email(cx, email=<email>, name=<name>)` (the existing helper, which `INSERT`s `(email, name, phone, source, created_at, updated_at)` if absent). Idempotent; returns the count created. (The hourly GHL sync only upserts — it does NOT prune — so these rows persist.)

### Component 2 — run the backfill on prod (console-gated)

`POST /api/console/backfill-affiliate-people` (`app.py`, console-key gated like the other `/api/console/*` ops): runs `backfill_affiliate_people`; supports `?dry_run=1` (report who WOULD be created, write nothing). Returns `{ok, created, emails}`. Triggered once at go-live (covers `vicsantos`).

### Component 3 — on-approval coverage (so future affiliates are covered)

When an affiliate becomes approved, ensure a `people` row:
- In `patch_affiliate` (`PATCH /api/affiliates/<id>`): after a status change to `approved`, call `customers.find_or_create_by_email(cx, email=<affiliate email>, name=<affiliate name>)`.
- In `/affiliate/apply` (where a signup is created — note its default `status` is `'approved'`): after the insert, call the same helper.
(Best-effort wrap so it never breaks the approve/apply flow.)

### Component 4 — retire/redirect the standalone portal + login

- `/affiliate/portal` → `302 /portal/login` (the standalone dashboard page is retired; affiliates use the personal portal, which shows their dashboard via 2b-1/2b-2).
- `/affiliate/login-request` (POST) → `302 /portal/login` (unify on the personal-portal magic-link; stop emailing the affiliate-specific magic link).
- `/affiliate/login-verify` (GET) → `302 /portal/login` (any in-flight affiliate magic link lands at the personal login instead of the retired page).
- The post-apply / social redirects that currently send users to `/affiliate/portal?token=…` (in `/affiliate/apply` and the social-links submit page redirect) → `/portal/login`.
- **Keep unchanged (public/needed):** `/affiliate` (apply + recruit landing), `/affiliate/hub/<slug>` (public recruit hub), `/affiliate/apply` (POST), `/affiliate/apply-form`, and `/affiliate/portal-data` (harmless data endpoint; the standalone page that consumed it is gone, but leaving the endpoint avoids breaking any lingering caller).

## Non-goals

- Removing the `affiliate-portal.html` file or the `/affiliate/portal-data` endpoint (leave them; the route just redirects away — lowest-risk retirement).
- Step 3 (full membership portal provisioning) — 2b-3's backfill is a targeted down-payment, not the general provisioning.
- Practitioner portal.

## Error handling

- `backfill_affiliate_people` none-raising per affiliate (a single bad row doesn't abort the batch).
- On-approval coverage is best-effort (try/except → never breaks approve/apply).
- Redirects are plain 302s.

## Testing

**Offline (tmp sqlite) — `backfill_affiliate_people`:**
1. Seed `affiliate_signups` (2 approved, 1 pending) + a `people` row for one approved email. Run → creates a `people` row only for the approved email missing one (count 1); the pending and the already-present are skipped. Re-run → 0 (idempotent).
2. The created `people` row has the affiliate's email + name (so self-login + greeting work).

(Components 2-4 are app.py routes/redirects — verified live.)

**Live post-deploy:**
3. `POST /api/console/backfill-affiliate-people?dry_run=1` → reports `vicsantos336699@` as the one missing; then without dry_run → `created: 1`; re-run → `created: 0`.
4. `/affiliate/portal`, `/affiliate/login-request`, `/affiliate/login-verify` → 302 to `/portal/login`.
5. Public surfaces unaffected: `GET /affiliate` and `/affiliate/hub/<a-slug>` still 200.
6. End-to-end: an approved affiliate (e.g. a test one in `people`) can `/portal/login` → `/portal/me` → see their ambassador dashboard (the destination the redirects point to). `vicsantos` now has a `people` row, so they can too.

## Rollout

Ships on merge → Render deploy. Go-live: run the backfill (dry-run then real), verify the redirects + public surfaces. Then the standalone affiliate portal is retired in favor of the personal portal. Step 3 (general provisioning) + the practitioner dashboard render remain.
