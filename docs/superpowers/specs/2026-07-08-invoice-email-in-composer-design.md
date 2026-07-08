# Invoice publish + email, in the composer

**Date:** 2026-07-08
**Context:** Automated client workflow for Rae. The "Hand off to Rae" button (#713)
already raises the invoice as a proposed order alongside the analysis draft. Rae
reviews + publishes the analysis in the composer (`/console/biofield-portal`). This
adds the invoice's review/edit/publish to that same page, plus an optional email so a
client who already saw their analysis hears that the invoice is ready.

## Goal

From the composer, for the loaded client, Rae can:
- see the raised invoice (total, lines, status),
- **edit** it (opens the order form, prefilled),
- **publish** it to the portal, with or without emailing the client.

One page for the whole handoff. Nothing is charged; publish only makes the pay card
visible and (optionally) sends a notice.

## Pieces

### 1. `GET /api/console/client-invoice?email=` (new, console-gated)
Latest non-cancelled order for the email. Returns:
```
{ok, order: {id, status, portal_published, pay_status, total_dollars,
             edit_url, lines: [{name, qty, amount_dollars}]}}   // or order: null
```
- `edit_url` = `/orders/new?edit_order=<id>&key=…` (the existing prefilled editor).
- Read-only; `null` when the client has no order.

### 2. `POST /api/console/order/<id>/publish-to-portal` — add `email` flag
Body `{email: true}`: after the existing publish (portal_published + invoice_token),
resolve the client's **portal link** (`ensure_token` + `portal_link`) and send a short
notice via `_send_full_report_email` (the analysis email path). Returns `emailed: bool`.
Email is best-effort: a send failure never fails the publish. Copy:

> Aloha [name],
> Your invoice is ready. You can view it and pay right on your healing home page:
> [portal link]
> With aloha, Dr. Glen & Rae

The link is the client's portal (shows analysis + the invoice pay card), matching the
analysis email — one destination, one link.

### 3. Composer Invoice panel (`console-biofield-portal.html`)
When a client email is loaded (Load existing / scan select / after analysis publish),
fetch `client-invoice` and render a card:
- total + per-line summary + a status chip (`to publish` / `published` / `paid`),
- **Edit invoice** → `edit_url` (new tab),
- **Publish invoice & email client** (bright yellow `.btn.notify`),
- **Publish invoice (no email)** (ghost),
- hidden entirely when the client has no raised order.
Re-fetch after publish so the chip updates. Buttons disable during the request.

## Editing before publish (the confirmation Rae asked about)
- Analysis: edited in the composer as today.
- Invoice: **Edit invoice** opens `/orders/new?edit_order=<id>` prefilled with lines +
  prices (change qty, add/remove remedies, adjust price, save); the order stays
  `proposed` and fully editable until she publishes.

## Tests
- `client-invoice` returns the latest order with lines + edit_url; `null` when none.
- publish-to-portal with `email:true` sends via a mocked `_send_full_report_email`
  and returns `emailed:true`; `email:false`/absent does not send.
- A send failure still returns `ok:true` (publish succeeded).

## Out of scope
- No new charging path (INVOICE_PAYLINK_ENABLED already governs pay).
- No auto-email on the handoff raise; email is Rae's explicit action at publish.
