# Spec 2a-3 — AI-suggested gift → console-approve → next order

**Date:** 2026-06-19
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`)
**Parent:** Spec 2a (reviews). Builds on 2a-2/2a-2b (merged). Next: 2b (referral coupons).

---

## Problem

A hero video review (the full 5 points) deserves a thank-you. 2a-3 has the AI suggest a fitting physical gift for such a reviewer, lets Glen/Rae approve/swap/reject it in the console, and delivers the approved gift as a free line on the person's next in-house order — with a "pending gift" indicator so it's never missed on other order paths.

## Scope (2a-3)

When a review reaches 5 points, the video worker asks the AI to pick a fitting gift from a configurable catalog (tailored by the review + the person's order history) and records it as **suggested**. The console shows the suggestion with Approve / Swap / Reject. An approved gift is **pending** for that email; the in-house order-entry path auto-adds it as a $0 line and marks it fulfilled, and the order-entry customer lookup surfaces a pending-gift indicator. Gated behind `REVIEWS_GIFTS` (default off, requires `REVIEWS_VIDEO`).

**Out of scope:** referral coupons (2b); auto-injecting gifts into self-serve/funnel order paths (only the in-house entry auto-adds; other paths show the indicator for manual add).

---

## Confirmed decisions (Glen, 2026-06-19)

- **Trigger:** a review whose total reaches **5 points** (only a strong video can — gifts are for hero UGC).
- **Cap:** **per qualifying review, but at most one gift per person per rolling 30 days** — at suggestion time, skip if the person has any **non-rejected** gift created in the prior 30 days (a rejected suggestion frees the slot).
- **Catalog:** a small **configurable catalog** (`data/review-gifts.json`) the AI picks from; seeded bamboo toothbrush / red nightlight / tuning fork.
- **AI suggestion is tailored:** the AI sees the review text/transcript + the product + the person's recent **order history** + the catalog.
- **Delivery:** pending-gift record per email; **auto-add a $0 line at in-house order entry** (`api_orders_manual`) + mark fulfilled; **console pending-gift indicator** for other paths.

---

## Architecture

### Catalog + store — `dashboard/review_gifts.py`
- Catalog loaded from `data/review-gifts.json` (`[{"sku","label","description"}]`): `load_catalog() -> list`, `catalog_by_sku() -> dict`, `valid_sku(sku) -> bool`.
- Table `review_gifts(id INTEGER PK, review_id INTEGER, email TEXT, gift_sku TEXT, gift_label TEXT, reason TEXT, status TEXT DEFAULT 'suggested', created_at TEXT, approved_by TEXT, approved_at TEXT, fulfilled_order_id INTEGER, fulfilled_at TEXT)`. `status` ∈ {suggested, approved, rejected, fulfilled}.
- Functions: `init_table`, `add_suggestion(cx, review_id, email, sku, label, reason) -> id`, `recent_active_gift(cx, email, days=30) -> bool` (any status != 'rejected' created within `days`), `get_for_review(cx, review_id) -> dict|None`, `set_status(cx, gift_id, status, by="")` (stamps approved_by/at on approve), `swap_sku(cx, gift_id, sku, label)`, `pending_for(cx, email) -> list` (status='approved', not fulfilled), `mark_fulfilled(cx, gift_id, order_id)`, `suggested_queue(cx) -> list` (status='suggested', newest first).

### AI suggestion — extend `dashboard/review_scoring.py`
`build_gift_prompt(review_text, product, order_history, catalog) -> (system, user)` (pure) + `suggest_gift(client, review_text, product, order_history, catalog, *, strip=lambda s: s) -> {"sku": str, "reason": str} | None`. The model picks one catalog `sku` and a short reason tailored to the review + the person's recent products. Fail-closed: any parse error, an empty catalog, or an invalid/missing sku → `None` (no suggestion). `reason` dash-stripped.

### Worker trigger (extend `_drain_review_videos`)
After the existing score/credit block, when `_REVIEWS_GIFTS` and the review `total == 5` and `review_gifts.get_for_review(rid)` is None and **not** `review_gifts.recent_active_gift(email, 30)`: gather `order_history` = recent product names via `orders.list_orders_by_email`, call `suggest_gift(_cl, transcript_or_body, product, order_history, catalog, strip=_strip_dash)`; if a valid sku comes back, `add_suggestion(...)` (status 'suggested'). Wrapped in its own try/except — a gift-suggestion failure never affects scoring/points/trim and never aborts the sweep. (`transcript` for video reviews; the review `body` otherwise — though only video reaches 5.)

### Console — `/console/reviews` + actions
- `GET /api/console/reviews` includes each review's gift (`gift_sku/label/reason/status`) when one exists. `static/console-reviews.html`: on a review with a suggested/approved gift, show "Suggested gift: <label> — <reason>" with **Approve / Swap / Reject** controls (Swap = a catalog dropdown). NO emoji.
- Dispatch-spine actions (`dashboard/reviews_actions.py`, RBAC `(OWNER, OPS)`, `LOW_WRITE`): `reviews.gift_approve {review_id, sku?}` (optional `sku` overrides the AI pick via `swap_sku` first, then `set_status('approved', by=actor)`), `reviews.gift_reject {review_id}` (`set_status('rejected')`). A `GET /api/console/gift-catalog` returns the catalog for the Swap dropdown.

### Redemption — in-house order entry
- In `api_orders_manual`, after `items_rec` is built and `total_cents` computed, when `_REVIEWS_GIFTS`: for each `review_gifts.pending_for(customer_email)`, append a line `{"slug": gift_sku, "name": gift_label + " (gift)", "qty": 1, "unit_price_cents": 0, "gift": True}` to `items_rec` (it does NOT change `subtotal`/`total_cents`). After `upsert_order` returns the order id, `mark_fulfilled(gift_id, order_id)` for each. Wrapped so a gift-redemption error never blocks order creation.
- The order-entry **customer lookup** (`/api/customers/search` or the customer-load path) includes `pending_gifts: [{label}]` for the email so whoever creates ANY order sees the pending gift (manual add on non-in-house paths).

### Flag
`_REVIEWS_GIFTS = os.environ.get("REVIEWS_GIFTS", ...)`. Gates the worker suggestion, the console gift UI/actions, and the redemption. Requires `REVIEWS_VIDEO`. Fully inert when off.

---

## Data flow
1. Video worker scores a review to 5 points → (flag on, within cap) AI suggests a catalog gift → `review_gifts` row `suggested`.
2. Glen opens `/console/reviews`, sees the suggestion, and **Approve** (or Swap then approve / Reject).
3. Approved gift is `pending_for(email)`.
4. Next in-house order for that email auto-adds a $0 gift line and marks the gift `fulfilled` (+ order id); other order paths show the pending-gift indicator for a manual add.

## Error handling
- `suggest_gift` fail-closed (no suggestion) — never blocks scoring/credit/trim.
- Worker gift step wrapped separately; the monthly cap + `get_for_review` guard prevent duplicates.
- Redemption wrapped so a gift error never blocks order creation; `pending_for` returns only approved-unfulfilled, and `mark_fulfilled` is keyed by gift id (idempotent — a fulfilled gift won't re-add).
- Invalid/edited catalog: `valid_sku` guards both the AI pick and a console Swap; an approved gift whose sku later leaves the catalog still carries its stored `gift_label` for the line item.
- Flag off → no suggestion, no console gift UI, no redemption.

## Testing
- **Catalog/store:** `load_catalog`/`valid_sku`; `add_suggestion`; `recent_active_gift` true within 30 days for non-rejected, false for rejected / outside window; `pending_for` (approved-unfulfilled only); `set_status`/`swap_sku`/`mark_fulfilled` transitions; `suggested_queue`.
- **`suggest_gift` (fake client):** returns a valid catalog sku + reason; invalid sku → None; empty catalog → None; never raises; reason dash-stripped.
- **Worker (mock transcribe/score/suggest):** a 5-point review within cap → a `suggested` gift row; a review under 5 → none; a person with a non-rejected gift in the last 30 days → none (cap); flag off → none; a suggestion failure leaves points/score intact.
- **Console actions:** `gift_approve` → approved (+ approved_by); `gift_approve` with `sku` swaps first; `gift_reject` → rejected; RBAC denies non-OWNER/OPS; catalog endpoint returns items.
- **Redemption:** an in-house order for an email with an approved gift adds a $0 line (subtotal/total unchanged) + marks the gift fulfilled with the order id; a second order does NOT re-add (fulfilled); flag off → no gift line; customer lookup surfaces `pending_gifts`.
- Follow deploy-chat test isolation (tmp `$DATA_DIR`; mock Supabase; importorskip playwright; `importlib.reload` + idempotent action registration). Console UI = manual visual pass. NO emoji; no em dashes in generated text.

## Flags
`REVIEWS_ENABLED` + `REVIEWS_VIDEO` + new **`REVIEWS_GIFTS`** (default off; requires `REVIEWS_VIDEO`). Inert when off.

## Notes
- Gifts attach to hero video reviews (only a video reaches 5), matching "physical gifts reserved for hero UGC video worth featuring."
- The $0 gift line never affects pricing/discount/shipping (price 0, excluded from subtotal). Rae still physically packs the item; the line + the indicator make it visible and tracked.
- Reuses the video worker, the Anthropic client, the dispatch spine + `/console/reviews`, the orders line-item structure, and `orders.list_orders_by_email`. No new external dependency.
