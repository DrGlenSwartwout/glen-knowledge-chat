# Portal — guarded findings backfill for existing portals

**Date:** 2026-07-07
**Depends on:** #661 (findings baked at publish time, forward-only). This backfills the portals published *before* #661.
**Scope:** one small prod endpoint + one local driver script + tests. No change to the publish path or the chip UI.

## Problem

#661 populates `findings` on the *next* publish of each client (forward-only). Portals
published before it keep `findings: []`, so their chips stay empty until re-published.
We want to fill them in — but a naive "re-publish everything" is a landmine:

- The publish path posts to `/admin/portal/upsert` with `send=True`, which **emails the
  client** even for an existing portal. A bulk re-publish would mass-email clients.
- Re-publishing rebuilds the *whole* content (layers, greeting, reorder pricing) from the
  local authored intake — so it can change more than findings, and needs each client's
  original special price (the local publish-tracking table isn't even present on this Mac).
- Only clients with a **local authored intake** can be rebuilt at all. On this Mac there
  are **8 intakes**; **3** have an intake email matching an existing portal (safe), **5**
  do not (a re-publish would create a *duplicate* portal under the intake email + email
  the client). The other ~180 prod portals have no local intake here.

## Goal

Fill `content.findings` on existing portals **without** touching any other content field,
**without** emailing anyone, and **without** ever creating a portal. Scan findings must be
computed locally (prod has no `e4l.db`) and pushed to prod surgically.

## Non-negotiable guards (the whole point of this slice)

1. **No email.** The new endpoint has no email code path at all.
2. **Never create.** The endpoint patches only an *existing* portal; unknown email → 404,
   never an insert.
3. **Findings-only.** The endpoint mutates exactly `content["findings"]` on the portal
   record and on matching biofield-report rows. Every other field is read and written back
   unchanged (read-modify-write of the stored JSON).
4. **Matched-email-only + has-findings.** The driver script targets only emails that already
   have a portal AND whose scan yields a non-empty findings set. It logs every skip.
5. **Idempotent.** Re-running patches the same value; safe to run repeatedly.
6. **Dry-run by default.** The script prints what it would patch; `--apply` executes.

## Design

### New prod endpoint — `POST /api/console/portal/backfill-findings`

Console-key gated (`_portal_console_ok`). Body: `{"email": str, "findings": [ {code,name,description,rank} ], "scan_date": optional str}`.

- Read the `client_portals` row for `email`. **If none → 404** (`{"ok": false, "found": false}`); never insert.
- Portal record: `content = json.loads(content_json)`, set `content["findings"] = findings`,
  write back the merged content (only `findings` changed).
- Biofield reports: for each `portal_biofield_reports` row for this email whose `scan_date`
  matches (or all rows if `scan_date` omitted), read its content JSON, set
  `content["findings"] = findings`, write it back via the existing `upsert_report`.
- No email, no token minting, no status change. Return
  `{"ok": true, "patched_portal": bool, "patched_reports": int}`.

Because the render reads *report* content when report dates exist and the *portal record*
content otherwise, patching both guarantees the chips light up regardless of which path the
client's portal uses.

### Local driver — `scripts/backfill_portal_findings.py`

Runs on the Mac (has `e4l.db` + the authored intakes + `CONSOLE_SECRET`).

1. Load the set of existing portal emails from `GET /api/console/portal-links` (matched-email guard).
2. For each authored intake email that is in that set:
   - Enumerate the client's report `scan_date`s (via `GET /api/portal/<token>` `scan_dates`,
     or the console links data). For each date, compute findings locally:
     `scan_context(email, scan_date).findings`, trimmed to `{code,name,description,rank}`.
   - Also compute a portal-record fallback (latest scan) for the no-`scan_date` patch.
   - Skip the client entirely if every computed findings list is empty; **log the skip**.
3. Dry-run (default): print, per client, the scan_date(s) and finding counts it *would* patch.
4. `--apply`: POST each to `/api/console/portal/backfill-findings`. Print the endpoint's
   `patched_*` result per client. Never touches an email absent from the portal set.

The script reuses the exact trim helper contract from #661 (`{code,name,description,rank}`)
so backfilled findings are byte-identical to freshly-published ones.

## Rejected alternative

**Full re-publish with `send=False`.** Simpler (no new endpoint) but rebuilds the entire
content from the local intake, so it can change layers/greeting/reorder pricing, and needs
each client's original special price (untracked here). It cannot guarantee "findings-only,"
which is the whole safety requirement. Rejected.

## Out of scope

- Portals with no local authored intake (~180 here) — unreachable from this Mac; a future
  run from the other Macs' intake DBs covers them with the same script + endpoint.
- The 5 email-mismatched local intakes — the matched-email guard skips them (re-publishing
  them belongs to an identity-merge cleanup, not this backfill; see [[feedback_identity_merge_review]]).
- Any UI or publish-path change.

## Verification

- **Endpoint unit tests** (pytest, in-process test client) in a new
  `tests/test_portal_backfill_findings.py`:
  1. Unknown email → 404, and no `client_portals` row created.
  2. Existing portal with `content.findings == []` + a report row → after the call, both the
     portal record content and the report content have the posted findings, and **every other
     content field is unchanged** (assert the full dict equals the original with only
     `findings` swapped).
  3. No email is sent (no send path exists — assert by construction / no email helper called).
  4. Idempotent: second call yields the same stored content.
- **Driver dry-run** (manual): run without `--apply`; confirm it lists only matched-email
  clients with non-empty findings and skips the rest with logged reasons. Confirm counts
  match the 3 safe local intakes.
- **Live (manual, `--apply` on the 3 safe clients):** after applying, fetch each portal's
  `/api/portal/<token>` and confirm `findings` is populated and non-other fields are intact;
  render one to see the chips.
