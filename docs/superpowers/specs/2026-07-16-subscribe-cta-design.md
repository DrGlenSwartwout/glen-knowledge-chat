# Shopper "Subscribe & Save" CTA — Design

**Date:** 2026-07-16
**Status:** Draft for review
**Repo:** deploy-chat
**Follows:** 2026-07-16-sellable-program-bundles-autoship-design.md (backend, shipped #936)

## Goal

Surface a shopper-facing **"Subscribe & save — save more each time"** option on the buy page for every autoship-eligible product, so a **paid member** can start an autoship subscription (card vaulted, recurring at the escalating ladder). Non-paid visitors are routed to become a member. This is the frontend the backend #936 was built for.

## Scope

- **Autoship-eligible products** — bundles with `autoship_eligible: true` (ladder 12→29) AND single-SKU **Functional Formulations** (`_qty_eligible` = `qty_pricing` and not `info_only`; ladder 3→25). Device bundles, non-FF single SKUs (ionizers/nightlights), and `info_only` show no CTA.
- **Paid-member benefit (model A):** the subscribe flow is for paid members; non-paid visitors get a "join to unlock" route to `/membership`.

## Background (what already exists — no change needed)

- `POST /reorder/subscribe` (`app.py:25382`): auth = signed-in `rm_reorder_email` cookie (401) + `is_member`/ToS (403 `need_optin`); requires `SUBSCRIPTIONS_ENABLED`, `PRICING_ENGINE_CHECKOUT`, Stripe active. Body `{items:[{slug,qty}], address:{name,street,city,state,zip,country}, cadence_months: 1|2|3}`. Prices via `_price_cart(subscriber_order_count=0, subscriber_active=True, …)`, creates a Stripe Checkout session with `save_card=True`, returns `{ok, stripe_url}`. Device bundles already rejected.
- Per-line ladder (`app._subscription_tier_resolver`): bundle line → `tier_for_bundle` (12→29), single SKU → `tier_for` (3→25); paid-gated on recurring.
- One-time buy pattern to mirror: `static/begin-buy.html` — inline address form (`#name,#ship-street,#ship-city,#ship-state,#ship-zip`), `placeOrder()` (`:662`) POSTs to `/begin/checkout/<slug>`, `renderConfirmation()` (`:741`) redirects to `data.stripe_url`. ToS gate via `OptinGate.show({onAgree: placeOrder})` (`:719`).
- Manage-existing portal `static/subscription.html` — cadence vocab `{1:'Monthly',2:'Every 2 months',3:'Quarterly'}` (`:113-116`). Match it.

## Design

### 1. Backend: expose the fields the CTA needs (`app.py::begin_product_data`, ~6336)

`/begin/product-data/<slug>` currently omits the flags the CTA must branch on. Add to its response dict (all source values already in scope in that function):

```python
data["autoship_eligible"] = bool(p.get("autoship_eligible")) and not p.get("info_only")
data["bundle"] = bool(p.get("bundle"))
_viewer_email = (get_authenticated_user(request) or {}).get("email") or request.cookies.get("rm_reorder_email", "")
data["is_paid_member"] = _is_paid_member(_viewer_email)
# ladder preview — server-authoritative, never hardcode 12/29 in JS
from dashboard import subscriptions as _subs
if p.get("bundle") and p.get("autoship_eligible"):
    data["autoship"] = {"first_pct": _subs.tier_for_bundle(0), "cap_pct": _subs.tier_for_bundle(99)}
elif p.get("autoship_eligible"):
    data["autoship"] = {"first_pct": _subs.tier_for(0), "cap_pct": _subs.tier_for(99)}
```

- `autoship_eligible` here folds in the `info_only` exclusion so the frontend has one boolean to gate on.
- `is_paid_member` reuses `_is_paid_member`; the viewer email is resolved exactly as elsewhere in the function.
- `autoship.first_pct`/`cap_pct` drive the copy so percentages stay truthful and centralized.

**No other backend change.** The subscribe endpoint, pricing, cron, and device gate are done.

### 2. Frontend: the CTA on `static/begin-buy.html`

The buy page fetches product-data on load. Based on the new fields, render one of three states below the one-time controls:

**State A — autoship-eligible + `is_paid_member: true`:** a purchase-mode toggle:
- `( ) One-time  ( ) Subscribe & save` (default One-time; Subscribe visually emphasized).
- When **Subscribe** is selected, reveal a cadence `<select>` (Monthly / Every 2 months / Quarterly, value 1/2/3, default 1) and a ladder line: *"Save {first_pct}% on your first shipment, climbing to {cap_pct}% — save more each time. Cancel anytime."*
- The existing **Place your order** button, when Subscribe mode is active, calls a new `placeSubscription()` that POSTs `{items:[{slug, qty}], address:{…the inline form…}, cadence_months}` to `/reorder/subscribe` and, on `{ok, stripe_url}`, does `window.location.href = stripe_url` (identical redirect to the one-time flow).
- Error handling: `401 not signed in` → show a "Sign in to set up Subscribe & Save" prompt (request the reorder magic link); `403 need_optin` → reuse `OptinGate.show({onAgree: placeSubscription})`; other errors → inline message.

**State B — autoship-eligible + `is_paid_member: false`:** a teaser card, no toggle:
- *"Members save {first_pct}–{cap_pct}% with Subscribe & Save on this program."* → button **"Become a member"** → `window.location.href = '/membership'`.

**State C — not autoship-eligible** (device bundle / info_only): render nothing (one-time buy only).

### 3. Copy

- Headline: **"Subscribe & save — save more each time"** (no em dash in shipped copy — use the hyphen-space form or a colon; no ALL CAPS).
- Cadence labels match the manage portal exactly: Monthly / Every 2 months / Quarterly.
- Percentages come only from `data.autoship.*` (server), never literals.

## Non-goals

- Managing/cancelling/skipping a subscription — already exists at `/subscription`.
- Any pricing/ladder/backend change — done in #936.
- Single-SKU vs bundle logic beyond the ladder percentages (both use the same CTA; the percentages differ by data).
- A CTA on `static/begin-product.html` (the content shell has no purchase controls; buying happens on begin-buy).

## Testing

- **Backend unit** (`tests/test_product_data_autoship_fields.py`, imports app → doppler): `/begin/product-data` for a bundle returns `autoship_eligible: true, bundle: true, autoship.first_pct == 12, cap_pct == 29`; for a single autoship SKU `first_pct == 3, cap_pct == 25`; for a device bundle `autoship_eligible: false`; `is_paid_member` reflects the viewer.
- **Frontend:** drive `begin-buy.html` headless (webapp-testing / render-verify) for a bundle as (a) paid member → toggle + cadence + ladder copy shows correct percentages and a Subscribe POST hits `/reorder/subscribe`; (b) non-paid → "Become a member" → `/membership`; (c) device bundle → no CTA.
- **End-to-end (staging/prod-verify):** a paid member subscribes to a bundle → redirected to Stripe → subscription row created at the bundle ladder. Verify via the manage portal / DB, not just the redirect.

## Rollout

- Verify **in prod** (not dev) that `SUBSCRIPTIONS_ENABLED`, `PRICING_ENGINE_CHECKOUT`, and Stripe-active are on; the CTA silently can't complete otherwise. `/membership` is behind `MEMBERSHIP_PRODUCTS_ENABLED` — confirm on, else route State B to `/begin/ascend`.
- deploy-chat is merge=deploy, no CI — render-verify the buy page live after deploy.

## Open items for review

1. **Subscribe POST auth fallback** — if a paid member lacks the `rm_reorder_email` cookie, the POST 401s; v1 shows a magic-link sign-in prompt. Acceptable, or should paid-member auth be unified first (larger)?
2. **State B destination** — `/membership` vs `/begin/ascend` (depends on `MEMBERSHIP_PRODUCTS_ENABLED` in prod).
