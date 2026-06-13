# Reorder cart (v1) — logged-in client reorder → Stripe

**Date:** 2026-06-13 · **Status:** approved, implementing

## Context

The Client Reorder/Retention nurture (Phase 3) links clients to `Truly.VIP/Reorder`.
Glen's vision: a returning client logs in, sees what they have ordered, and
reorders in a few taps. This is the build behind that link. Decisions (with Glen):
**magic-link identity**, **reuse the existing Stripe retail checkout** (bypasses
GrooveKart's broken checkout), and **phase it** — v1 is the reorder cart; v2 adds
"continue shopping" (catalog browse). It's a logged-in client cart seeded from
order history, not a full storefront yet.

## Flow
`Truly.VIP/Reorder` → `/reorder` → enter email → magic link → one click back,
authenticated → cart page → checkout → Stripe → order recorded.

## Components
- **`GET /reorder`** — page with two states: email-entry (unauthenticated) and the cart (authenticated). Static `reorder.html` + JS, matching the existing funnel pages.
- **Magic-link identify** — reuse `/auth/magic-link/request` + `/auth/magic-link/verify` (app.py:3098/3136) with **`purpose=reorder`**; verify sets a short-lived reorder session (same mechanism the other portals use) that the `/api/reorder/*` calls authenticate against. Unknown email still returns "check your inbox" (no existence reveal). 15-min token TTL.
- **`GET /api/reorder/items?scope=last|all`** (authed) — reads the client's `orders` by email, parses `items_json`, and returns products as `{name, slug, current_price, last_ordered, available}`. `scope=last` = items from the most recent order; `scope=all` = deduped across all orders. `slug` via the existing `_resolve_buy_slug(name)`; `current_price` + `available` from the products catalog (same source `/begin/buy/<slug>` uses); discontinued products → `available:false`.
- **`POST /reorder/checkout`** (authed) — body `[{slug, qty}]`. Build a retail order `out` (reuse the order-builder behind `/begin/checkout/<slug>`): resolve each slug → current price, line = price×qty, sum → total; then `stripe_pay.create_checkout_session(total_cents, customer_email, …, metadata={kind:"reorder", items:[…]})` (one session for the cart total, matching `_stripe_checkout_url_for_retail`). Return the Stripe URL. The existing checkout-return handler (`/begin/checkout-return`) records the order + payment; extend it to write the reorder order's items into the `orders` row (so future reorders see them).
- **Repoint** `Truly.VIP/Reorder` (Rebrandly) → `https://illtowell.com/reorder`.

## The cart page (authenticated)
- **"Your last order"** — each item: checkbox + quantity stepper; a "select all / reorder all" toggle.
- **"See everything you've ordered"** → loads `scope=all`: full deduped product history, each with checkbox + quantity + last-ordered date; discontinued shown disabled.
- **Cart summary** — selected items × qty, line totals, subtotal.
- **"Checkout (N)"** → POST to `/reorder/checkout` → redirect to Stripe.
- **Cart state is client-side** (selections + quantities live on the one page until checkout). No server-side cart until v2 (continue-shopping, where you leave and return).

## Out of scope (v2+)
"Continue shopping" / catalog browse + add; server-side persistent cart;
auto-reship / subscriptions; per-line address editing (Stripe collects shipping);
any GrooveKart path.

## Edge cases
No past orders → friendly empty state + store link. Discontinued product → shown,
not checkout-able. Price changed since last order → current price (the UI notes
"current price"). Expired/used token → re-request. Quantity min 1. Cart with only
unavailable items → checkout disabled.

## Open implementation details (resolve at build)
- **`items_json` shape** — confirm what each item carries (name? slug? qty? price?) to map → product + current price.
- **The retail order-builder** behind `/begin/checkout/<slug>` (the code that produces `out` with total) — factor a small reusable helper that takes `[{slug, qty}]` and returns the order `out`, used by both the single-slug funnel checkout and `/reorder/checkout`.
- **Reorder session mechanism** — confirm how `/auth/magic-link/verify` hands a session to subsequent API calls (cookie vs token) and reuse it for the `/api/reorder/*` auth.

## Tests
- Magic-link: issue + verify for `purpose=reorder`; unknown email reveals nothing; expired token rejected.
- `/api/reorder/items`: dedupes by email across orders; resolves slug + current price; flags discontinued; `scope=last` vs `all`.
- `/reorder/checkout`: builds the correct total from `[{slug, qty}]` × current price; unavailable slug excluded/blocked; empty cart rejected; returns a Stripe URL (stub Stripe in tests).
- Auth gate: `/api/reorder/*` and `/reorder/checkout` reject without a valid reorder session.

## Verification
`/reorder` → request a magic link for a test client with order history → verify →
items list shows their products with current prices → select a couple + set
quantities → checkout → Stripe session opens for the correct total → return path
records the order.
