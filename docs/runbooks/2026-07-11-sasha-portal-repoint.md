# Runbook — Sasha portal repoint to the infoceutical scan (Phase 0)

**Date:** 2026-07-11 · **Owner:** Glen/Claude · **Type:** operational (prod data), no app code

This is the immediate task that motivated the portal scan-history feature. It repoints Karin's family portal from the older FF-based analysis (Sasha's July 2 scan, scan_id 1037250) to the newer infoceutical-based scan (July 9 seed, scan_id 1035975). It runs against **prod data** and is gated on resolving an identity question first. It uses the console set-current endpoint shipped in Task 4 (`POST /api/console/portal/set-current`).

Background: Sasha is Karin's cat. The July 2 scan is superseded because it still recommends Functional Formulations rather than infoceuticals. See the vault memories `feedback_biofield_ff_vs_infoceutical_currency`, `reference_portal_seed_email_vs_contact`.

## Preconditions

- The scan-history branch (`sess/cbf8f69f`, PR #799) is deployed to prod so `/api/console/portal/set-current` exists. (If repointing before merge, use a guarded direct write to `client_portals.content_json.current_scan_date` instead — same effect, one field.)
- Console access: `X-Console-Key` header (or `?key=`) = `CONSOLE_SECRET`.

## Steps

### 1. Resolve identity (do this before touching anything)
Determine, in prod:
- Which email Karin's portal token `HwyyJEumbmoPHznsiB7UKsKblA6Haqc6dCQrt4AYp3E` resolves to (`client_portals` row).
- Which `(email, scan_date)` the infoceutical scan (scan_id 1035975) lives under in `portal_biofield_reports` — the cat login `permanentlyyours777@hawaiiantel.net` vs Karin's mailbox `permanentlyyours@hawaii.rr.com`.

Two outcomes:
- **Same email** as the portal → a plain `set-current` on that email works.
- **Different subject/email** (e.g. scan is under a member email) → the portal reaches it via the household `?member=<email>` mechanism, and "current" is set on the *member's* `client_portals` row, not the primary's. Confirm the `household.can_view(primary, member)` relationship exists first.

### 2. Ensure the scan is ingested in prod
If scan_id 1035975 has no `portal_biofield_reports` row in prod, push it first:
- `e4l-scan-manifest-push.py` then `e4l-scan-recommendations-push.py` (doppler prd).
- These 502 during a deploy window — retry after the deploy settles (see memory `feedback_e4l_ingest_needs_prod_push`).

### 3. Repoint
Set `current_scan_date` to the infoceutical scan's date for the resolved email:

```
curl -s -X POST "$PROD/api/console/portal/set-current" \
  -H "X-Console-Key: $CONSOLE_SECRET" -H "Content-Type: application/json" \
  -d '{"email":"<resolved-email>","scan_date":"<infoceutical-scan-date>"}'
```

Expected `{"ok": true}`. A `400 no report for that scan_date` means step 2 (ingest) is incomplete — do not force it; the guard is doing its job.

### 4. Render-verify (mandatory — payload correctness is not enough)
Open the live portal in headless Chrome (memory `feedback_render_the_page_not_the_payload`):
- Current Analysis shows the infoceutical scan (Heart Health/immune/mitochondrial layers, not the FF Muscle Mass/etc.).
- Scan History lists both dates, with the infoceutical one badged Current.

### 5. Notify — later, and only by hand
Do NOT auto-send. Karin's "new analysis is ready" email is a separate confirm-to-send (`POST /api/console/portal/notify-scan`) once `PORTAL_SCAN_NOTIFY_ENABLED` and the bulk (GHL-v2/Mailgun) channel are live in prod. Until then, no notification fires.
