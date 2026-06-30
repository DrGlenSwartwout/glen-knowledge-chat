# Order card collapse/expand — design

## Context
The console Orders board (`/console/orders`, `console-orders.html`) renders each order as
a card that always shows products, fulfillment, and action buttons — visually heavy. Glen
wants each card collapsed by default to a compact header, expanding on click to reveal the
products and everything else.

## Behavior (confirmed with Glen)
- **Collapsed (default):** a clickable header only:
  - Line 1: name (or email) + payment-status badge (Paid / claimed / Unpaid) + backorder badge.
  - Line 2 (meta): **date placed** · **where/how placed** (friendly source label, + "· Pickup"
    when channel is pickup) · **amount**.
  - A ▸ chevron affordance (rotates to ▾ when open).
- **Expanded (click the header):** reveals the product list, tracking, biofield-report link,
  the "Fulfill lines" panel, and the action buttons.
- Per-card state, default collapsed. Action buttons live in the body, so clicking them does
  not toggle the card (only the header toggles).
- A deep link `?order=<id>` auto-expands and flashes that card.

## Implementation (single file: `static/console-orders.html`)
- Restructure `cardHtml(o)` into `.card > .card-head (clickable) + .card-body (hidden)`.
- CSS: `.card-body{display:none}`, `.card.open .card-body{display:block}`, a `.chev` that
  rotates when `.card.open`; `.card-head{cursor:pointer}`.
- `toggleCard(headEl)` toggles `.open` on the parent card (bound to `.card-head` onclick).
- New helpers: `dateStr(iso)` → "Jun 28, 2026"; `whereLabel(o)` → friendly source
  (in-house→"In-house (phone/email)", portal-reorder→"Portal reorder", reorder→"Reorder",
  funnel→"Website", biofield_trial→"Biofield trial") + "· Pickup" when `channel==='pickup'`.
- The relative "age" text + its red `.old` styling are replaced by the date placed. The
  lane already conveys fulfillment status, so the header status is the payment badge.
- `maybeHighlight()` adds `.open` to the deep-linked card before scrolling/flashing.

## Out of scope
- No backend/API change (the card already receives `created_at`, `source`, `channel`,
  `total_cents`, `pay_status`, `items`).
- No change to actions, fulfillment logic, or the lanes.

## Verification
Render the board in a headless browser with seeded orders across lanes: confirm cards are
collapsed by default (header shows date · where/how · amount + name + pay badge, no products
visible), clicking the header expands to show products + actions + fulfillment, clicking an
action button does not collapse the card, a `?order=<id>` deep link opens + flashes the card,
and zero console errors.
