# In-House Order Entry — "Pickup (no shipping)" Option

**Date:** 2026-06-26
**Status:** Approved (design)
**Author:** Glen + Claude

## Problem

Local clients pick up their order in person, so it should carry **$0 shipping**. The in-house order builder (`/api/orders/manual`) auto-computes shipping from cart geometry with no way to suppress it, so a pickup order is over-billed by the shipping amount.

## Goal

A "Pickup (no shipping)" checkbox on in-house order entry that creates the order with `shipping_cents = 0` and marks it as a pickup, so the board, the hosted invoice page, and record-payment all reflect no shipping.

## Design

The orders table already has a `channel` column (default `'retail'`) and a stored `shipping_cents` — **no schema change**. Pickup is persisted as `channel='pickup'`; shipping is zeroed wherever an in-house order's shipping is *computed*.

Exploration confirmed there are exactly **two** such sites for an in-house order (every other `create_invoice`/`_price_cart` call is a funnel/subscription/cron checkout, never pickup, and the hosted invoice page + board read the order's **stored** `shipping_cents`):
1. **Order creation** — `/api/orders/manual` computes `shipping_cents` via `_price_cart` and stores it.
2. **Invoice edit** — `/api/invoice/<token>/update` recomputes `shipping_cents` for an existing order.

### Components

1. **`dashboard/orders.py` — pure helper (offline-tested):**
   `effective_shipping_cents(pickup, computed_cents) -> int` — returns `0` when `pickup` is truthy, else `int(computed_cents or 0)`. Single source of the pickup-shipping rule, used by both routes.

2. **`/api/orders/manual` (`app.py`, ~24237):**
   - read `pickup = bool(body.get("pickup"))`.
   - `shipping_cents = effective_shipping_cents(pickup, pc.get("shipping_cents"))` (the existing fallback `shipping_cents, get_cents = 0, 0` on pricing error stays).
   - pass `channel="pickup" if pickup else "retail"` to `upsert_order` (it already takes `channel`).
   - `total_cents` math is unchanged (it already adds `shipping_cents`, now 0).

3. **`/api/invoice/<token>/update` (`app.py`, ~24437):**
   - after the recompute, `shipping_cents = effective_shipping_cents((order.get("channel") or "") == "pickup", pc.get("shipping_cents"))` so editing a pickup order never re-adds shipping. (Fetch the order's channel from the order this route already loads by token.)

4. **`static/order-new.html`:**
   - a "Pickup (no shipping)" checkbox; when checked, include `pickup: true` in the `/api/orders/manual` POST body. Reflect "$0 shipping — pickup" in any client-side total hint.

### Why this is complete (no QBO send-invoice change)

In-house orders carry their pricing in the stored order row. `_invoice_summary` (the hosted invoice page) reads `order.shipping_cents` directly; the board reads the stored order; record-payment marks the stored total paid. The direct `qb.create_invoice(... + _shipping_line(...))` sites are all funnel/subscription/cron flows (`biofield_checkout`, `begin_checkout`, `_checkout_cart`, founding/reorder subscribe, charge cron) — not in-house board orders — so none recompute shipping for a pickup order. Persisting `channel='pickup'` means any future path can honor it.

## Error handling

- Helper is pure/none-raising (treats non-int `computed_cents` via `int(... or 0)`).
- The existing pricing-fallback in `/api/orders/manual` (`shipping_cents, get_cents = 0, 0`) is unaffected; a pickup order with a pricing error still ends at 0 shipping (consistent).

## Testing

**Offline (pytest) — the helper:**
1. `effective_shipping_cents(True, 1299) == 0`; `effective_shipping_cents(True, 0) == 0`.
2. `effective_shipping_cents(False, 1299) == 1299`; `effective_shipping_cents(False, None) == 0`.

**Live post-deploy (`app.py`/HTML can't import offline):**
3. `POST /api/orders/manual` with `pickup:true` (console-keyed) → created order has `shipping_cents == 0`, `channel == "pickup"`, `total == subtotal − discount`; same lines without `pickup` show the normal shipping.
4. The board card for the pickup order shows no shipping; the hosted invoice page total has $0 shipping.
5. Editing the pickup order via `/api/invoice/<token>/update` keeps `shipping_cents == 0`.
6. `order-new.html`: the checkbox posts `pickup:true` and the created order is $0 shipping.

## Rollout

Ships on merge → Render deploy. Then Karin's order: `biofield-analysis` @ $100 + 5 remedies @ $50, **Pickup checked** → $350, no shipping; Record payment → Check $350 when she pays.
