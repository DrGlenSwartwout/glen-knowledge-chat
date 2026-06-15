# Certification Biofield bonus (Mechanic 3)

A committed certification enrollee earns a bonus Causal Biofield Analysis **each month for 12
months**, plus **one per module completed** (1–12). Each earned bonus is delivered
concierge-style: it records an entitlement and drops an ops task for the team to schedule/run.
Ships dark behind `CERT_BONUS_ENABLED`.

## Commitment (admin-set, no in-app cert checkout)
Certification is a booked/manual sale, so commitment is recorded by an admin action:
`POST /api/cert/commitment {email, kind, started_at}` (console-key gated), where `kind` is
`pif` or `monthly12`. `{email, clear:true}` ends it. Stored in the sqlite `cert_commitments`
table (`dashboard/cert_bonus.py`). Both kinds currently earn the same 12-month schedule.

## What's granted
A daily cron `POST /api/cron/biofield-bonuses` (X-Cron-Secret) sweeps active commitments:
- reads the practitioner's `modules_completed` (`practitioner_portal.modules_completed_for_email`, Supabase);
- `due_bonuses()` computes owed grants: **monthly** 1..min(months elapsed since `started_at`, 12),
  and **level** 1..min(modules_completed, 12); minus anything already granted (the
  `cert_bonus_grants` ledger, PK `(email, kind, idx)`);
- for each new grant: inserts one `todos` row (category `biofield-bonus`,
  dedup_key `cert-bonus-<email>-<kind>-<idx>`, owner `glen`) and records the grant.
Idempotent (ledger + dedup_key); `?dry_run=1` counts without writing.

## Why concierge, not the self-serve gate
The #114 `biofield_readiness` table is one row per email (a single self-serve purchase), so it
can't represent repeated monthly/level bonuses. Cert enrollees are high-touch (a $3,600+ program
with Glen), so each bonus becomes an ops task the team runs. Auto-seeding the self-serve gate for
a cert bonus is a possible later option.

## Flags / ops
- `CERT_BONUS_ENABLED` (default off) gates the sweep (the admin commitment endpoint is console-gated regardless).
- **Scheduling: built in.** The sweep (`_run_biofield_bonuses`) runs **daily in-process at 15:00 UTC (5am HST)** via the app's `BackgroundScheduler` (registered in `_start_scheduler`, alongside the hourly console push) — no external cron needed. It's flag-gated, so it's a harmless no-op until `CERT_BONUS_ENABLED` is on + there is ≥1 cert commitment.
- The same logic is also exposed at `POST /api/cron/biofield-bonuses` (X-Cron-Secret; `?dry_run=1`) for manual/on-demand runs and verification.

## Deferred
PIF vs 12-month behavioral differences (same schedule for now); practitioner notification beyond
the ops task; auto-seeding the self-serve Biofield gate; a Supabase-native commitment field.
