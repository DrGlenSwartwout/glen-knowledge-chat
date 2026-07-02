# Reveal ordering flow — trust + honesty fixes (SP-A)

**Date:** 2026-07-02
**Driver:** Steve Fox feedback — tried to order his Remedy Matches, was "brought directly to a payment screen" with no shipping/context (felt sketchy, bailed), came back, and got "no longer available." Two real problems in the reveal → order flow. This is SP-A of the reveal→$1-unlock→order redesign; SP-B ($1 = lifetime free-level membership) is a separate follow-up.

## Root cause (confirmed on prod)
- Steve = reveal #42, `sfnase@hotmail.com`, **not approved** (`first_approved=False`) → `_biofield_visible_slugs` returns **[]**. `POST /begin/biofield/<token>/order-checkout` filters the cart against the empty visible-set and returns **"Your cart is empty or those items are no longer available."** (`app.py:2815`). The message is a lie — the real state is "not unlocked."
- The cart's **"Order my remedies"** button jumps **straight to `window.location = stripe_url`** (`begin-biofield.html:819`) with no in-app shipping/review — the bare-payment-screen distrust. The per-remedy **"Order"** button links to `/begin/buy/<slug>` (also straight to Stripe).

## A1 — Stop lying; route locked users to unlock (not a failing order)
**Server (`app.py`, `begin_biofield_order_checkout` ~2810):** distinguish two cases when the filtered `items` is empty:
- Caller sent items but all were filtered out by `visible` → `{"ok": False, "need_unlock": True, "error": "These matches aren't unlocked yet — unlock your full analysis to order."}` (400).
- Caller sent no items → `{"ok": False, "error": "Your cart is empty."}` (400).

**Frontend (`begin-biofield.html`):**
- On a `need_unlock` response, don't `alert()` — scroll to and pulse the existing unlock affordance (the free-reveal button / `$1` CTA), so a locked client is guided to unlock instead of hitting a dead end.
- Defensive: the "Order my remedies" bar already hides when the cart is empty; also treat a `need_unlock` result as "not orderable yet."

## A2 — In-app cart + shipping review before Stripe
Insert a **review step** between "Order my remedies" and Stripe, on the reveal page (branded, not a bare Stripe screen):
- A panel/modal listing the cart items + total (reuse the `order-preview` numbers already fetched), plus a **shipping form**: name, street, city, state, zip (country default US), pre-filled when known.
- **Prefill source:** add a `ship_prefill` object (`{name, street, city, state, zip}`) to the reveal payload in `begin_biofield_reveal` (the client's own stored address via the same resolver `order-checkout` uses), so the form is pre-populated and the client just confirms. It's the client's own data on their own token — no new PII exposure.
- **Confirm → `order-checkout` WITH `address`** (the endpoint already accepts `body.address` and runs it through `_resolve_ship_address`) → then redirect to `stripe_url`. So the address is captured/reviewed in-app first; Stripe is the final payment step only.
- Route the per-remedy **"Order"** button into this same flow (add the remedy to the cart + open the review panel) instead of `/begin/buy/<slug>` straight-to-Stripe, so *all* ordering is reviewed. (Keep `/begin/buy/<slug>` route intact for any other linkers; only the reveal-page button changes.)

## Non-goals (SP-B / later)
- The `$1` lifetime free-level membership grant (duration change + copy) — separate PR, needs Glen's sign-off since it edits today's Model #2.
- Turning on `BIOFIELD_TRIAL_ENABLED` (the `$1` offer) — Glen's flip.
- Reworking `/begin/buy/<slug>` itself.

## Error handling / edge cases
- `order-preview` failure → the review panel still opens with items; total shows "—"; the client can still submit (server re-prices authoritatively).
- Missing/partial prefill → empty editable fields; required fields validated client-side before enabling "Continue to payment."
- `need_optin` (ToS) response is unchanged (still surfaced).
- Address is advisory to `_resolve_ship_address` (server remains authoritative); an empty address still works (Stripe/stored fallback), but the form nudges completion.

## Testing
- Server: the `need_unlock` branch is a small pure-ish change; covered by reasoning + the existing biofield-cart tests (`tests/test_biofield_cart.py`) which exercise visible-slug filtering — extend one to assert `need_unlock` on the all-filtered case if importable, else reason.
- Frontend: inline `<script>` passes `node --check`; the review panel + prefill verified by reasoning and a live smoke after deploy (drive an approved reveal: add → review shows prefilled address → checkout reaches Stripe).
- Live acceptance: an unpaid/locked reveal now shows the unlock CTA on order attempt (no "no longer available"); an unlocked reveal orders through the in-app review step.
