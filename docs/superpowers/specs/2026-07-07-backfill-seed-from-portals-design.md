# Backfill driver — seed from portals (cover all portals with a scan)

**Date:** 2026-07-07
**Follow-up to:** #664 (backfill endpoint + driver). Broadens the driver's reach.
**Scope:** the local driver script `scripts/backfill_portal_findings.py` + its tests. No endpoint change, no prod-app change.

## Problem

The #664 driver seeds its candidate list from the local authored intakes
(`biofield_auth_tests`, 5 on this Mac), so a run only reaches portals whose client was
authored here — 3 of 185. But the findings come from `e4l.db` (keyed by email), and this
Mac IS the canonical E4L scan store (ingestion runs here): 914 client scans, matching
**119 of the 185 portals**. The intake seed was an over-conservative choice, not a real
constraint — it left ~116 backfillable portals untouched.

Measured:

| | count |
|---|---|
| Prod portals (has_token) | 185 |
| Clients in this Mac's `e4l.db` | 914 |
| Portals whose client has a scan here | **119** |
| Reachable by current (intake-seeded) driver | 3 |

## Goal

Backfill every portal that has a matching E4L scan on this Mac (119), safely, from the
portal list — dropping the intake-seed limitation.

## Design

In `scripts/backfill_portal_findings.py`:

- **Seed from portals.** In `main()`, set the candidate list to the tokened portal emails
  (already fetched for the `portal_emails` guard): `candidate_emails = sorted(portal_emails)`.
  Remove the `biofield_auth_tests` query, the `INTAKE_DB`/`sqlite3` intake read, and the
  `BIOFIELD_DB` env dependency.
- **Rename** `plan_backfill(portal_emails, intake_emails, ...)` → `(portal_emails,
  candidate_emails, ...)`. Pure-function logic is unchanged: it still skips a candidate not
  in `portal_emails` ("no existing portal") — trivially never fires now — and skips a
  candidate whose computed findings are all empty ("no findings computed"), which is exactly
  the ~66 portals with no scan here.
- **`--limit N`** (optional): after sorting, cap `candidate_emails` to the first N before
  planning, so the first `--apply` can be a small, verifiable batch. Default: no limit.
- **Per-client resilience:** wrap each candidate's gather-and-patch work in `try/except` that
  logs the email + error and continues, so one client's network/HTTP blip can't abort a
  119-client run. (Also resolves the Minor logged in #664's review.)

Everything else stays: the console endpoint `POST /api/console/portal/backfill-findings`,
the trim to `{code, name, description, rank}`, `findings_for_scan_date` for dated report rows
+ `scan_context` (latest) for the portal-record patch, dry-run by default, idempotent.

## Safety (unchanged, still sufficient)

Seeding from portals does not weaken any guard:
- **Never creates:** the endpoint 404s an email with no portal; seeding from the portal list
  means every candidate already has one, so this is moot but still enforced server-side.
- **No email:** the endpoint has no send path.
- **Findings-only:** read-modify-write of just `content.findings`.
- **Identity-safe:** findings are computed for the portal's OWN email via
  `findings_for_scan_date(portal_email, scan_date)`; a client whose scans live under a
  different email simply computes `[]` and is skipped — never patched with someone else's or a
  mismatched scan's findings.

## Out of scope

- The 66 portals with no E4L scan on this Mac (client never scanned, or scan under a different
  email) — genuinely un-backfillable until they have a scan; no code covers them.
- Any endpoint / UI / publish-path change.

## Verification

- **Unit (plain pytest):** update the 4 `plan_backfill` tests in `tests/test_backfill_driver.py`
  to pass `candidate_emails=` (rename) — same four assertions (skip-no-portal, per-date,
  portal-record, skip-no-findings) still hold. `--limit` is a trivial `candidate_emails[:N]`
  slice applied in `main()` before the pure call; it needs no dedicated unit test (the sliced
  list flows through the already-tested planner) and is exercised by the live dry-run below.
- **Live dry-run (manual):** run without `--apply`; confirm it now plans ~119 patches and
  skips the no-scan portals with "no findings computed" (no "no existing portal" skips, since
  the seed is the portal list). Redact emails in any shared output.
- **Live apply (manual, batched):** `--apply --limit 10` first, verify a couple of those
  portals carry findings, then run the rest.
