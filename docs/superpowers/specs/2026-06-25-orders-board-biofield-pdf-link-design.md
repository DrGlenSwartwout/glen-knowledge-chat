# Orders Board — "Biofield report (PDF)" Print Link

**Date:** 2026-06-25
**Status:** Approved (design)
**Author:** Glen + Claude
**Parent:** builds on the portal-publish work (PR #320/#321). The publish stores each client's report PDF URL in `portal_biofield_reports` content (`report_pdf.url`).

## Problem

When Rae packs a shipment on the console Orders board (`/console/orders`), she wants to include a printed copy of the client's Biofield Analysis. Today the PDF lives only on the client's portal / Glen's Mac — Rae has no one-click way to print it at pack time.

## Goal

On the Orders board, any order whose client has a **published (confirmed)** biofield report shows a "🖨 Biofield report (PDF)" link that opens the PDF in a new tab to print. Reuses the `report_pdf.url` the publish already stores. Read-only; **deploys to prod**.

## Non-goals

- Auto-printing / folding into the packing-slip flow (a larger future option).
- Surfacing draft/blurred reports (only `status == "confirmed"`).
- Token-gating the asset download (the PDF stays the opaque `/portal-asset` URL — same trust model as the existing assets).
- Any change to order data, fulfillment, or the publish flow.

## Design

### 1. Lookup helper (`dashboard/portal_biofield_reports.py`, pure, offline-tested)

`report_pdf_urls(cx, emails) -> dict[str, str]` — given an iterable of emails, returns `{email_lower: url}` for each email whose **latest confirmed** report carries a non-empty `content.report_pdf.url`. Emails with no confirmed report, or a confirmed report lacking a pdf url, are omitted. Implementation: one query over `portal_biofield_reports WHERE lower(email) IN (...) AND status='confirmed' ORDER BY scan_date DESC`, parse `content_json`, keep the first (latest) per email that has a `report_pdf.url`. Empty/zero-length `emails` → `{}` (no query). None-raising.

### 2. `/api/orders` GET annotation (`app.py`, prod, live-verified)

In the GET branch of `bos_orders_create` (after the existing name-backfill block), add a grouped annotation mirroring that block's pattern (try/except, skip-on-error, one query — no per-order round-trips):
- collect `emails = {o["email"].strip().lower() for o in rows if o.get("email")}`
- `urls = portal_biofield_reports.report_pdf_urls(cx, emails)`
- for each order, set `o["biofield_pdf_url"] = urls.get((o.get("email") or "").strip().lower(), "")`

(The module is already importable in app.py — confirm the import alias; `portal_biofield_reports` is used by the `/api/portal` routes.)

### 3. Orders board render (`static/console-orders.html`, prod, live-verified)

In `cardHtml(o)`, inside the `meta` div (after the tracking line, before the `</div>` that closes meta), add — only when present:
```javascript
+ (o.biofield_pdf_url?'<br><a href="'+esc(o.biofield_pdf_url)+'" target="_blank" rel="noopener">&#128424; Biofield report (PDF)</a>':'')
```
(`&#128424;` = 🖨. Uses the file's existing `esc()` helper.)

## Error handling

- Helper is none-raising (bad JSON → skip that row).
- The `/api/orders` annotation is wrapped in try/except that logs and continues (orders still render without the link), exactly like the adjacent fulfillment/name-backfill annotations.

## Testing

**Offline (tmp sqlite) — the helper:**
1. `report_pdf_urls` returns `{email: url}` for an email whose latest confirmed report has `content.report_pdf.url`.
2. Picks the LATEST confirmed report when an email has several scan_dates.
3. Omits an email whose only report is non-confirmed (e.g. `ai_draft`), and one whose confirmed report has no `report_pdf`.
4. Lowercases emails on input and in the returned keys; empty `emails` → `{}` with no query.

**Live post-deploy (`app.py` can't import offline):**
5. `curl /api/orders` (console-keyed) for a client who has a published confirmed report → that order's JSON includes a non-empty `biofield_pdf_url`; an order for a client with no report has `""`.
6. Load `/console/orders` and confirm the "🖨 Biofield report (PDF)" link renders on that order and opens the PDF (200) in a new tab.

## Rollout

Ships on merge → Render deploy. Verifiable end-to-end once a client (Karin) has been published. No data migration.
