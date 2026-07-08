# Current-report selection + idempotent hand-off

**Date:** 2026-07-08
**Context:** A client's portal showed a stale AI **reveal** biofield report instead of
the manually authored one. Root cause (from a full read of the portal model): the
display prefers per-scan `portal_biofield_reports` rows over `client_portals.content_json`
and picks the **latest by scan_date**, with **no current/active concept**. The reveal
publish path writes a per-scan row (with a scan_date); the hand-off pushed **without**
a scan_date, so it only touched `content_json` — which the display ignores once a
per-scan reveal row exists. Separately, each hand-off click raised a **new** invoice
(Steve accumulated #51–#55).

## Fixes (this PR — the urgent correctness half)

### 1. Current-report pointer
- `content.current_scan_date` marks the client's current report.
- Display (`api_client_portal`): pick `?scan_date=` if requested, else
  `content.current_scan_date` if present, else newest-by-date (unchanged fallback).
  So a manual Biofield wins over a stale reveal **regardless of date**; live portals
  without the pointer behave exactly as before (forward-only).
- The hand-off now passes the test's `scan_date` (or today) → writes a per-scan report
  AND stamps `content.current_scan_date` (`default_handoff_push`).
- The composer publish also stamps `content.current_scan_date = scan_date`, so
  publishing a report makes it current.

### 2. Idempotent hand-off (update, not pile up)
- `_cancel_open_handoff_orders(cx, email)` cancels a client's OPEN drafts (proposed,
  unpaid, `portal_published=0`); published/paid orders are left alone.
- `POST /api/orders/manual` honors `replace_open: true` — cancels those drafts before
  creating the fresh order, returns `cancelled: [ids]`.
- `biofield_invoice.default_create_order(..., replace_open=True)` and the hand-off
  route pass it, so a repeated hand-off **replaces** its prior draft. Result surfaces
  as `invoice.replaced` on the hand-off response.

## Follow-up (next PR)
- **History tab** on the portal: older scan reports (already addressable via
  `scan_dates` + `?scan_date=`) and past invoices (extend `_published_invoices_for`
  with a paid/terminal variant) filed to a reference view; current stays on the main.
- Steve cleanup: after deploy, one hand-off makes the manual Biofield current and
  collapses his open drafts; cancel the published dups (#51/#52) on the Orders board.

## Tests
- Portal shows the `current_scan_date` report even when a later-dated reveal report exists.
- `_cancel_open_handoff_orders` cancels open unpublished proposed drafts, keeps published/paid.
- Existing hand-off/invoice/publish suites green.
