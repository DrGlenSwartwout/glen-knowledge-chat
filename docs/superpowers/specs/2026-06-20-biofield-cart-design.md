# Begin #4c — Biofield Reveal Match-to-Order Cart

**Date:** 2026-06-20
**Status:** Approved (design); ready for implementation plan
**Repo:** deploy-chat (Flask, Render `glen-knowledge-chat`, illtowell.com)
**Parent:** Begin-page redesign #4. #4a (PR #194) renders the matched remedies on the token-verified Biofield reveal; #4b (PR #197) added the $1 unlock into the paid tier so a paid member sees the full matched set. #4c is the "match-order polish": turn that matched set into a single cart that checks out in one Stripe session.

---

## Problem

The Biofield reveal lists each matched remedy with its own single-item "Order" button -> `/begin/buy/<slug>`. A member matched to several remedies has to run a separate checkout per remedy, and never sees the volume / mix-and-match discount that ordering the set together would earn. There is already a multi-item cart checkout (`/reorder/checkout`) that prices a whole cart through the pricing engine, creates one QBO invoice, and mints one Stripe session — but it is keyed to the reorder magic-link cookie, so it is not reachable from the token-verified reveal.

## Goal

Let a member order their matched set from the reveal as one cart: pick which matched remedies plus quantities, see the live volume-discounted total, and check out once — reusing the existing cart-checkout engine. Behind `BIOFIELD_CART_ENABLED` (default off), dark until a visual pass, matching #4a/#4b.

## Scope (#4c)

Two new token-scoped, member-facing endpoints on the reveal (`order-preview`, `order-checkout`), a DRY extraction of the cart-checkout core out of `reorder_checkout` so both callers share it, the reveal front-end cart affordances (checkbox + qty stepper per remedy + a sticky total bar), and the `cart_enabled` flag in the reveal payload.

**Out of scope:** the `/begin/match` conversational-chat surface (a later sub-project — #4c covers the Biofield reveal only); points-redemption or referral-code entry in the reveal cart (the extracted helper accepts them, but the bar passes `0`/none in v1); any change to the $1 trial (#4b), the per-remedy single Order button (stays), the billing/pricing engine, or the `/begin/checkout-return` handler (reused as `kind=reorder`, unchanged).

---

## Confirmed decisions (Glen, 2026-06-20)

- **Surface: the Biofield reveal only** (the conversational match chat is a later follow-on).
- **Interaction model: select + quantities, then one checkout** — each matched remedy gets a pre-checked checkbox and a qty stepper; a sticky "Order N remedies — $total" bar shows the live volume-discounted total and goes to one Stripe checkout.
- **Cart operates on whatever is visible to that member** — paid members get the full matched set; a free member with only their top remedy unlocked gets the same bar with that one item. Blurred/withheld remedies are never in the cart (their slugs are never sent to the client, and the server rejects them).
- **Behind `BIOFIELD_CART_ENABLED`** (default off, dark until flipped). Checkout additionally still requires the existing `PRICING_ENGINE_CHECKOUT` + `STRIPE_ACTIVE` + membership.
- No emoji, no em dashes.

---

## Architecture

### The reuse decision (identity + DRY)
The reveal identifies the member by the **biofield token + their email** (`_biofield_verify_token` -> the `biofield_reveals` row), not the reorder magic-link cookie that `/reorder/checkout` keys off. So #4c gets its own token-scoped entry points but reuses the cart core. To avoid duplicating that core, extract it from `reorder_checkout`.

### 1. Extract `_checkout_cart(email, cart, *, ship, points_to_redeem_cents=0, referral_code=None)`
Pull the engine-path body of `reorder_checkout` (app.py ~10472-10522) into a helper that: caps the requested points redemption to balance, resolves the referral coupon, `_price_cart(cart, ship=..., coupon_pct=..., points_to_redeem_cents=...)`, rejects an empty priced cart, `qb.find_or_create_customer` + `qb.create_invoice`, `_ingest_order(source="reorder", channel="retail", ...)`, `_record_referral_if_any`, then builds `out` (invoice_id / doc_number / customer_id / total) and `stripe_url = _stripe_checkout_url_for_reorder(out, email) if _STRIPE_ACTIVE else ""`. Returns `{"out": out, "stripe_url": stripe_url}`. Raises `CheckoutError` for a pricing problem (caller maps to 400). `reorder_checkout` is rewritten to call this helper (behavior unchanged — its existing tests stay green). The new endpoint calls the same helper.

### 2. `POST /begin/biofield/<token>/order-preview`
Body `{items:[{slug, qty}]}`. Guard `BIOFIELD_CART_ENABLED` (else `{ok:false}`). Verify the token -> resolve `email` from the `biofield_reveals` row (bad token -> 404). Recompute the member's **visible matched set** server-side (the same gating `begin_biofield_reveal` uses: paid -> all remedies; free -> the unlocked top remedy if approved; else none) and reject any submitted slug not in it. Clamp each qty to 1..99. `ship` = the member's last order address on file or `{}`. `_price_cart(items, ship=ship)` -> return `{ok:true, lines:[{slug, qty, unit_cents, line_cents, list_cents, savings_cents}], subtotal_cents, shipping_cents, total_cents, savings_cents}`. Never raises (catch -> `{ok:false}`); used to drive the sticky bar's live total.

### 3. `POST /begin/biofield/<token>/order-checkout`
Body `{items, address?}`. Guard `BIOFIELD_CART_ENABLED` (else `{ok:false}`). Verify the token -> resolve `email`. Membership-gate: `if not is_member(amg_session, email): return {ok:false, need_optin:true}, 403`. Validate slugs are in the visible matched set (drop the rest); if the cart is then empty -> 400 "Your cart is empty or those items are no longer available." `ship` = request `address` or the last order on file. `_checkout_cart(email, items, ship=ship)` -> return `{ok:true, stripe_url, **out}`. Wrapped, never 500s. Because `_stripe_checkout_url_for_reorder` stamps `metadata.kind="reorder"`, the existing `/begin/checkout-return?kind=reorder` records the order + payment with no new handler.

### 4. Reveal payload + front-end (`static/begin-biofield.html`)
`begin_biofield_reveal` adds `cart_enabled = BIOFIELD_CART_ENABLED` to the payload. When true, the remedy render (paid full set, or the free member's single unlocked top remedy) adds per remedy: a pre-checked **checkbox** and a **qty stepper** (default 1), alongside the existing name-link + per-remedy Order button (both stay). A **sticky footer bar** "Order N remedies — $total" debounce-calls `order-preview` on any checkbox/qty change (rendering subtotal + shipping + total, and a "you save $X ordering together" line when `savings_cents > 0`), and on click calls `order-checkout` -> `location.href = stripe_url`. All dynamic text via `textContent`/`setAttribute`; no `innerHTML` of dynamic data. When `cart_enabled` is false, none of these render (current #4a/#4b behavior).

### Reuse / untouched
- Pricing/checkout: `_price_cart`, `qb.find_or_create_customer`/`create_invoice`, `_ingest_order`, `_record_referral_if_any`, `_stripe_checkout_url_for_reorder`, `_shipping_line`, `CheckoutError` — all reused via `_checkout_cart`.
- The reveal store/route/token (`_biofield_verify_token`, `biofield_reveals`), the visible-set gating in `begin_biofield_reveal`, `is_member`, the `/begin/checkout-return` handler (`kind=reorder`) — reused.
- Untouched: the $1 trial (#4b), the per-remedy single Order button, the billing/pricing engine, the daily charge cron, the journey map.

---

## Data flow
1. `BIOFIELD_CART_ENABLED` on. Member opens their approved reveal; payload carries `cart_enabled:true`; the page renders checkboxes + steppers + the sticky bar over the visible remedies.
2. Select / qty change -> debounced `order-preview` -> bar shows the live volume-priced subtotal + shipping + total (+ savings line).
3. Click the bar -> `order-checkout` -> `_checkout_cart` prices + creates the invoice + ingests the order + mints the Stripe URL -> `location.href = stripe_url`.
4. Stripe return -> existing `/begin/checkout-return?kind=reorder` records the order + payment -> member back on the confirmation.

## Error handling
- Flag off OR Stripe inactive -> payload omits the cart; both endpoints return `{ok:false}` (404 for an unknown token). No charge path.
- Non-member -> 403 `{need_optin:true}` (the existing OptinGate pattern).
- A submitted slug not in the member's visible set -> dropped; if that empties the cart -> 400 "empty / no longer available."
- `order-preview` never raises (catch -> `{ok:false}`); `order-checkout` is wrapped, never 500s; empty cart -> 400.
- Anti-bypass: the visible-set is recomputed server-side on every call; a free member can never cart a blurred deep remedy (its slug was never sent and the server rejects it).
- No new idempotency surface — Stripe's one-session-per-checkout is the dedupe, same as `/reorder/checkout` today.

## Testing
`tests/test_biofield_cart.py`, plus a regression pass on the reorder suite for the DRY extraction.
- **preview:** a valid visible set -> priced lines + subtotal + total; a slug outside the visible set -> rejected; flag off -> `{ok:false}`.
- **checkout:** member + visible set -> `_checkout_cart` invoked, returns a Stripe URL (mock `stripe_pay` + `qb`); non-member -> 403 `need_optin`; empty / all-invisible cart -> 400; flag off -> dark.
- **anti-bypass:** a free member (top-only visible) submitting a deep-remedy slug -> rejected by both endpoints.
- **DRY:** `reorder_checkout` still works through the extracted `_checkout_cart` (the existing reorder tests stay green — run that suite).
- Stripe / qb / GHL mocked throughout; tmp `LOG_DB`; init the biofield_reveals + memberships + subscriptions tables. No emoji; no em dashes.
- Front-end (the bar, the steppers, the redirect) = manual visual pass.

## Notes
- **Behind `BIOFIELD_CART_ENABLED` (default off).** Merge is dark; go-live = flip the flag in Doppler (with `PRICING_ENGINE_CHECKOUT` + `STRIPE_ACTIVE` already on). No new money path — the cart rides the engine that already powers `/reorder/checkout` in prod.
- All copy (the bar, the savings line) is provisional — BNSN pass later.
- The `/begin/match` conversational-chat cart is the natural next increment; #4c's `_checkout_cart` helper + token-scoped pattern set it up.
