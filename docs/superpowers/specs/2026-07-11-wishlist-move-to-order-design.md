# Wishlist → Move-to-Order (fold in quantities) — Design

**Date:** 2026-07-11
**Feature flag:** reuses `WISHLIST_ENABLED` (no new flag — this is a v2 of the shipped wishlist, #822)
**Builds on:** [[project_wishlist]] (#822), the portal reorder module (Task 6b), `/api/portal/<token>/checkout`

---

## Goal

Let a portal customer move items from their "Your wishlist" card into a real order — selecting which items, setting a quantity per item, and checking out the selected items together in one Stripe checkout, at their member (portal) price.

## Why this shape

The portal already has the exact machinery:

- **Reorder module** (`.remitem` rows in `static/client-portal.html`): each row has a quantity stepper (`.qty-ctl` / `data-qty`) and an "Order" button that posts `{items:[{slug, qty}]}` to `/api/portal/<token>/checkout`, which returns a live Stripe URL. Add-then-confirm — the charge only happens if the customer confirms on Stripe's hosted page.
- **Checkout endpoint** (`api_client_portal_checkout`, app.py ~18359): accepts `{items:[{slug, qty}]}`, clamps qty to 1..99, drops any client-posted price, and prices every line server-side via `_portal_priced_lines` (member pricing preserved). It validates every posted slug against `entitled = _portal_entitled_slugs(email)`, **unioned with `_accepted_recommendation_slugs(email)`** — on the stated logic that "an ACCEPTED doctor recommendation IS the authorization to buy."

A wishlist item is the same class of authorization, arguably stronger: the customer saved it themselves and is explicitly choosing to order it. So the entitlement gate is extended to also union the customer's own wishlist slugs.

## Architecture

Two changes, both small and both mirroring existing patterns.

### 1. Backend — extend the checkout entitlement union (app.py)

In `api_client_portal_checkout`, where the posted-items branch builds `entitled`, add a third union source: the acting email's own wishlist slugs. Flag-guarded by `WISHLIST_ENABLED` and wrapped in try/except so a failure can never break checkout (identical guard style to the accepted-recommendation union already there).

New pure helper in `dashboard/wishlist.py`:

```python
def slugs_for(cx, owner):
    """Set of slugs currently on `owner`'s wishlist. owner is the resolved
    'email:<addr>' / 'sess:<id>' key (see resolve_owner). Empty set if none."""
    rows = cx.execute("SELECT slug FROM wishlist WHERE owner = ?", (owner,)).fetchall()
    return {r[0] for r in rows}
```

In the endpoint (posted-items branch only), after the accepted-recommendation union:

```python
if _WISHLIST_ENABLED:
    try:
        with sqlite3.connect(LOG_DB) as _wcx:
            _wishlist.init_wishlist_table(_wcx)
            entitled = entitled | _wishlist.slugs_for(_wcx, "email:" + email)
    except Exception:
        pass  # wishlist-entitlement union failure must never break checkout
```

`email` in this endpoint is already the portal token's email, lowercased — the correct owner key (see [[feedback_portal_token_identity]]). The `?member=` household re-point is out of scope: the reorder module's per-row checkout does not re-point either, and a household caregiver ordering from a member's wishlist is not a v2 requirement.

**No new pricing path.** Wishlist items flow through the same `_portal_priced_lines` as every other line. A wishlist item the customer has never purchased is a first-time purchase at member price — which is correct: a portal customer is a member and gets member pricing.

### 2. Frontend — orderable wishlist rows + combined checkout (static/client-portal.html)

The "Your wishlist" card (`#wishlistCard`, built from `d.wishlist`) changes from remove-only to orderable:

- Each `.wishitem` row gains:
  - a **select checkbox** (unchecked by default),
  - a **quantity stepper** reusing the existing `.qty-ctl` / `.qtybtn` / `.qtyval` markup and the existing `data-qty` convention,
  - the existing remove (×) button (unchanged).
- Below the rows: a single **"Order selected (N)"** button, disabled when nothing is selected, showing the live count of checked items.
- Clicking it collects every checked row's `{slug, qty}` into one array and posts `{items:[...]}` to `/api/portal/<token>/checkout` (one checkout for all selected items), then redirects to the returned `stripe_url`. Same add-then-confirm flow, same one-shot disabled latch, same error surface as `reorderItem`.

Quantity is **order-time only** — read from the DOM stepper at click time. Not persisted to the wishlist table (no schema change, no migration). The wishlist is "save for later"; quantity is an ordering decision.

Ordered items **stay on the wishlist**. The checkout is add-then-confirm; we don't know at POST time whether the customer completes payment on Stripe. Once the order ingests it appears in "Your Remedies" (purchase history) on the next portal load, and the manual × remove already exists. No auto-remove.

The qty-stepper click handler currently scopes to `.remitem` rows (`qb.closest(".remitem")`). It will be generalized to also handle `.wishitem` rows (scope to the nearest row container that carries a `.qtyval`), so one handler drives both modules.

## Data flow

1. Customer opens portal → `api_client_portal` returns `payload["wishlist"]` (already built per-member after the `?member=` re-point).
2. Customer checks two wishlist items, sets quantities (2 and 1), clicks "Order selected (2)".
3. Frontend posts `{items:[{slug:"a", qty:2}, {slug:"b", qty:1}]}` to `/api/portal/<token>/checkout`.
4. Endpoint resolves `email` from the token, builds `entitled` = purchase-history ∪ accepted-recs ∪ **wishlist slugs**; both posted slugs pass.
5. `_portal_priced_lines` prices them at member price; QBO invoice + Stripe URL created exactly as for a reorder.
6. Frontend redirects to Stripe; customer confirms → order ingests as `source="portal-reorder"`.

## Error handling

- Endpoint: unchanged. Unknown slug (not in the unioned entitled set) → 400 "That item isn't available to reorder." Empty/invalid items → 400. Stripe inactive → 503. All existing.
- Frontend: reuse the `reorderItem` error pattern — a per-card error line, button re-enabled on failure, one-shot disabled latch to prevent double-fire (this codebase has been bitten by double-fire; see the reorder module's latch).
- Wishlist-union failure server-side is swallowed (try/except) — checkout still works for already-entitled items; a wishlist-only item would then 400, which is a safe degrade (no charge, clear message).

## Testing

**`dashboard/wishlist.py` — `slugs_for` (unit):**
- returns the set of slugs for an owner with multiple saved items
- returns empty set for an owner with none
- scopes by owner (owner A's slugs never include owner B's)

**Endpoint entitlement union (unit/integration around `api_client_portal_checkout`):**
- with `WISHLIST_ENABLED` on, a posted slug that is ONLY on the customer's wishlist (not in purchase history, not an accepted rec) passes the gate (mock `_portal_priced_lines` / Stripe boundary; assert no 400 "isn't available")
- with the same slug NOT on the wishlist and not otherwise entitled → still 400 (the union adds only wishlist slugs, nothing broader)
- union keyed off the **token's** email, not any begin-side identity
- union failure (wishlist table read raises) does not break checkout for an already-entitled slug

**Frontend (manual render-verify, documented in the plan):** anonymous save → portal load → select two items, set quantities → "Order selected (2)" posts one combined `{items:[...]}` → Stripe redirect. Verified locally against a real portal token before flipping nothing (flag already live).

## Out of scope (deferred)

- Persisting quantity on the wishlist row (schema change).
- Auto-removing ordered items from the wishlist (needs checkout-return plumbing to know payment completed).
- `?member=` household re-point on the wishlist order path (matches the reorder module, which also doesn't).
- Sharing / gift lists; back-in-stock / price-drop notifications.

## Global constraints

- Reuse existing `WISHLIST_ENABLED`; no new flag. The feature is dark wherever the wishlist card is dark.
- Never trust client-posted price; server-side `_portal_priced_lines` only (unchanged).
- Portal writes/reads key off the **token's** email, never begin-side cookies ([[feedback_portal_token_identity]]).
- Every DB access uses a dedicated `sqlite3.connect(LOG_DB)` connection; do not re-acquire `_db_lock` where a caller already holds it (non-reentrant).
- Copy rules: no em dashes, no ALL CAPS, structure/function framing only.
