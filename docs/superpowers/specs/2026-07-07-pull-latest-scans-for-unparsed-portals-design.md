# Pull + parse latest scans for the 56 unparsed portal clients

**Date:** 2026-07-07
**Follow-up to:** #669 (backfill seed-from-portals). Recovers the 56 portals whose scan is in the E4L manifest but not yet parsed.
**Scope:** one small flag on `02 Skills/scrape-e4l-http.py` (+ test) and an operational runbook. No deploy-chat app change.

## Problem

After the seed-from-portals backfill, 56 portals still have empty findings because their
latest E4L scan is a **manifest-only** record (no results) — `refresh-e4l-scans.py` recorded
that the scan exists but never downloaded/parsed the PDF (the PDF endpoint is the slow,
~84%-error, no-concurrency one; parsing is on-demand). See
[[reference_e4l_scans_manifest_vs_parsed]]. Their findings are recoverable by pulling +
parsing the scan, then re-running the backfill.

## Decisions (locked)

- **Latest scan only** — one PDF per client (~56 downloads), not full history. Bounds load on
  the flaky endpoint.
- **No reveal drafts, no vectorize** — only get scan *results* into `e4l.db` so the backfill
  lights up chips. (Reveal-draft synthesis and Pinecone vectorization are separate, optional.)

## Design

### Code: `--latest-only` on `02 Skills/scrape-e4l-http.py`

`scrape-e4l-http.py --clients <csv> --workers 1` already batches multiple clients in one auth
session with resume (skips already-downloaded via `e4l-scans-log.json`) — but it downloads
*every* scan per client. Add a `--latest-only` flag that, per client, reduces the scan list to
the single most-recent scan before the download loop.

- Add a pure helper `_latest_only(scans) -> list` (returns a 0/1-element list):
  `max(scans, key=lambda sc: (_parse_date(sc["date"]), int(sc["id"])))` wrapped in a list;
  `[]` when `scans` is empty. `_parse_date` handles `M/D/YYYY` and `YYYY-MM-DD`, returning a
  sentinel min-date for `"unknown"`/unparseable so a dated scan always wins; the `int(id)`
  tiebreak makes it robust even when all dates are unparseable (scan ids are monotonic).
- In `main()`, add `--latest-only` (action store_true). Right after
  `scans = get_client_scans(s, cid)`, `if args.latest_only: scans = _latest_only(scans)`
  (before the `is_downloaded` filter, so resume still applies to that one scan).
- Unit test (pure, no network) in `02 Skills/tests/`: `_latest_only` picks the newest by date,
  falls back to max id on messy/unknown dates, and returns `[]` for an empty list.

### Runbook (operational — run with Glen's go; writes to real client portals)

1. **Select the 56 client_ids** — portal clients whose latest e4l scan has no results. Compute
   from the portal list (`/api/console/portal-links`) ∩ `e4l.db`, producing a comma-separated
   client_id list (one per client, the most-recent client_id of the merged identity). A short
   documented python snippet; emails never printed.
2. **Scrape (the long, flaky step):**
   `scrape-e4l-http.py --clients <csv> --latest-only` (workers stays 1). Re-run until the
   download log shows all 56 present — resume skips successes, so each pass only re-attempts
   the failures. Partial success is fine; the pipeline is re-runnable.
3. **Parse:** `parse-e4l-scans.py` — parses newly-downloaded PDFs in `~/e4l-scans/` into
   `e4l.db` (`e4l_scan_results`), recording any per-PDF failures in `e4l_parse_failures`.
4. **Backfill:** `backfill_portal_findings.py --apply` — idempotent; the newly-parsed clients
   now yield findings and get patched (findings-only, no email, no create).
5. **Verify:** spot-check a few of the newly-parsed portals' `/api/portal/<token>` — findings
   populated; and report how many of the 56 landed vs still-failed (endpoint flakiness) so a
   later pass can pick up the remainder.

## Safety / expectations

- The scrape hits an external system (E4L portal) and is **slow + flaky** (~84% 500s,
  sequential). Expect multiple passes; some of the 56 may not download in a given run — that's
  expected, not a failure. Everything downstream is idempotent and re-runnable.
- The backfill step's guards are unchanged (findings-only, no email, never-create). Only the
  portals whose scan actually parsed get patched.
- Auth to the E4L portal is handled inside `scrape-e4l-http.py` (its existing
  `get_session_cookies`); no new credentials.

## Out of scope

- Reveal-draft synthesis and Pinecone vectorization for these scans (separate, optional).
- The 17 no-scan + 66 not-in-E4L portals (no scan exists to pull).

## Verification

- **Unit (pure):** the `_latest_only` test above (newest-by-date, id fallback, empty).
- **Operational:** the runbook's step 5 — a real spot-check that pulled+parsed portals now
  carry findings, plus a landed/failed count over the 56.
