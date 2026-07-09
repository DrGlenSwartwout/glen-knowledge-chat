# Per-client pickup default, and shippability the operator can see

**Date:** 2026-07-08 (rebased onto #734 on 2026-07-09)
**Status:** Implemented

## History — read this first

This spec was originally written to make "this order has nothing to ship" a derived
fact rather than a stored flag. While it was being implemented, **PR #734 shipped
that same core fix independently**: `dashboard/shipping.py:is_shippable(product)`
and the `_price_cart` box-count gate are on `main` and are NOT part of this branch.

#734 also decided that a **biofield hand-off is not a pickup** — it removed
`pickup: True` from `dashboard/biofield_invoice.py`, so hand-off invoices now bill
USPS on the physical remedy lines while the analysis fee (a service) contributes no
bottle. This branch does not touch that decision.

What remains here is the work #734 did not do.

## What this branch adds

### 1. The US-only address check follows shippability

`_price_cart` rejected a non-US country before it knew whether the cart contained
anything shippable (`app.py`, the guard formerly directly after the `country`
assignment). A client buying only a Biofield Analysis got "We ship to US addresses
only" — on a service.

The `country` assignment stays where it is; the `raise` moves to after the cart loop
and fires only when `box_counts` is non-empty:

```python
# US-only shipping — but only a cart with something to ship has an opinion
# about the address. An overseas client buying a service prices fine.
if box_counts and country not in ("US", "USA", ""):
    raise CheckoutError("We ship to US addresses only — please use a US forwarding address.")
```

Because `box_counts` is populated only by shippable lines (#734's gate), the two
decisions derive from one structure: a service can never suppress the box count yet
still trip the address check.

**Consequence, and it is public.** `app.py`'s biofield checkout catches
`CheckoutError` and returns a 400. Overseas visitors could not buy a Biofield
Analysis on the website. Now they can. An overseas cart containing a bottle still
raises.

### 2. The operator sees the derived state

`dashboard/products.py`'s `catalog()` emits `shippable` per product, computed by
`shipping.is_shippable` — the same predicate the pricer uses, not a reimplementation.

`static/order-new.html` uses it: when no cart line is shippable, the Pickup checkbox
is **disabled** and reads *"No shipping — nothing physical in this order"*, and the
Shipping row shows `$0.00`. The checkbox stops being something the operator must
remember to tick and becomes a readout.

An unknown slug is treated as shippable (`p ? p.shippable !== false : true`), matching
`is_shippable(None) == True` on the server. Both fail toward charging shipping.

### 3. Per-client pickup default

New `dashboard/client_prefs.py`, shaped like its sibling `client_prices.py`: a table
keyed by normalized email, one column `pickup_default`, and `get_pickup_default` /
`set_pickup_default`. Unknown email returns `False`. Setting twice leaves one row.

**Nothing writes it except an explicit operator toggle.** Saving an order does not.
`/api/orders/manual` does not. A per-order Pickup tick is an override for that order
alone — which is the whole design, and it is why the module has no
"record what this order did" helper.

In `static/order-new.html`, a second checkbox sits beside Pickup — *"Pickup by default
for this client"* — visible only when a customer email is present **and** the order has
at least one shippable line. An order with nothing to ship represents no fulfillment
decision and cannot teach us one. It POSTs immediately to the owner-gated
`/api/console/client-prefs`, on its own, not as part of the order payload.

Applying it:
- **New order** — selecting or typing a customer email reads the preference and
  pre-checks Pickup. Untick it and it stays unticked *for that order only*.
- **Edit** — the stored order's `channel` wins. `loadPickupDefault` will not touch
  `$("pickup").checked` when `EDIT_OID` is set.

A POST that omits `pickup_default` returns **400** rather than silently writing
`False`. Clearing a client's preference must be deliberate; a partial or retried
request must not do it by accident. An explicit `false` still flips it off.

## Testing

**Money path.** `tests/test_price_cart_shippable.py` pins that a service line never
reaches the box-fit call. Comparing invoice cents cannot detect the bug: at today's
rates the buggy qty-fallback (5 items → M) and the correct box-fit (4 bottles → M)
both return $23.00. The test therefore **spies on the `box_counts` handed to
`shipping.quote()`** and asserts the service slug is absent. Verified by removing the
gate and watching it go red.

**Country.** An overseas services-only cart prices without raising; an overseas cart
containing a bottle still raises `CheckoutError`.

**The preference.** Default false; round-trip; idempotent re-set; email normalized;
and the test that guards the design — `app.py` contains exactly one call to
`set_pickup_default`, and neither `createInvoice()` nor `editInvoice()` sends
`pickup_default`. `tests/test_client_prefs_route.py` exercises the route over real
HTTP (owner and non-owner, both verbs, the 400s), against a temp `LOG_DB`.

**Catalog tests are hermetic.** They monkeypatch `load_products`, because other tests
in the suite replace the global catalog and an earlier version of these tests passed
alone and failed in the full suite.

**Browser.** The order builder was driven through all four states, including: tick the
client default, reload, confirm Pickup pre-checks, untick Pickup for that order, then
confirm via the API that `pickup_default` is still `true`.

## Known, out of scope

Nothing here changes how the orders board labels a pickup, or the biofield hand-off
shipping decision made in #734.
