# Derived shipping + per-client pickup default

**Date:** 2026-07-08
**Status:** Approved design, ready for implementation plan

## Problem

An order that contains nothing physical still quotes USPS shipping.

`_price_cart` (`app.py:5364-5367`) adds every cart line to `box_counts` and
`total_bottles` regardless of what the line is:

```python
bt = _shipping.resolve_bottle_type(slug, p)
box_counts[bt] = box_counts.get(bt, 0) + qty
total_bottles += qty
```

`resolve_bottle_type` (`dashboard/shipping.py:559`) falls through to `"default"`
for a product whose `bottle_type` is null, which is exactly what the two service
SKUs (`biofield-analysis`, `evox-session`) have. So a Biofield-Analysis-only
order built in the console picks a Small Flat Rate box and charges for it.

Two consequences follow from this, and both are live today:

1. **Mixed orders bypass the box-fit catalog entirely.** A service resolves to
   bottle type `"default"`, which is not a real type in the capacity matrix, so
   `_shipping.quote()` returns `Unknown bottle type: default` for any cart that
   mixes a service with bottles. `_shipping_for_cart` (`app.py:5310-5323`)
   swallows that error and falls through to `_fallback_shipping_cents`, the crude
   quantity rule. Mixed orders have not been priced by the flat-rate catalog at
   all — they've been priced by the fail-safe.

   (Measured 2026-07-08: for four bottles plus an analysis, both paths happen to
   return $23.00 today, so no historical overcharge is demonstrable on that cart.
   The coincidence is not the design. A rate or capacity change moves the two
   paths apart without warning.)
2. **`pickup` is being used as a workaround.** `dashboard/biofield_invoice.py:119`
   hardcodes `"pickup": True` on the hand-off invoice purely to suppress the
   phantom shipping on the fee line. Those invoices then appear as pickups on the
   orders board (`static/console-orders.html:210`) and are excluded from combined
   shipments (`dashboard/combined_shipments.py:74`), even though Rae mails them.

Today `pickup` is the *only* mechanism for "no shipping." There is no
`no_shipping` column, no `digital` product field, and no per-client fulfillment
preference anywhere in the codebase.

## Design

### 1. Shippability is derived, never stored

Add one predicate to `dashboard/shipping.py`, beside `resolve_bottle_type`:

```python
def is_shippable(product) -> bool:
    """False for anything with no physical thing to put in a box."""
    p = product or {}
    return not (p.get("service") or p.get("info_only"))
```

This is the only place the question is asked. When a genuinely digital SKU
appears, it is taught here and nowhere else. (There is no `digital` field on
products today; `service` and `info_only` are the two real markers, carried by
`biofield-analysis`, `evox-session`, and `emf`.)

`_price_cart` consults it before contributing a line to `box_counts` and
`total_bottles`. A non-shippable line contributes to neither.

An order made entirely of non-shippable lines therefore reaches
`_shipping_for_cart` with an empty `box_counts`, which already returns `0` at
`app.py:5320`. No new code path — the existing empty-cart guard is simply reached
by the orders that deserve it. Every caller inherits the fix, because the console
in-house pricer, checkout, and invoice edit all route through `_price_cart`.

**Nothing is stored.** Add a bottle to a services-only order and shipping
reappears on the next price computation, because it was never a flag. This is
what satisfies "don't remember this setting for other orders" — there is nothing
to remember.

`effective_shipping_cents(pickup, computed)` (`dashboard/orders.py:973`) is
unchanged. Pickup still zeroes shipping; it just stops being the only way there.

### 2. The country check follows shippability

`_price_cart` rejects a non-US country outright (`app.py:5350-5352`), and
`_price_inhouse_invoice` neutralizes the country for pickups (`app.py:32688`)
solely to route around it.

The check now fires only when the cart has at least one shippable line. An
overseas client buying only a Biofield Analysis prices without error. The pickup
neutralization at 32688 stays as-is — it is still correct for a pickup order of
physical goods.

### 3. The operator sees the derived state

`/api/products` (`dashboard/products.py:60`) gains a `shippable` key per product,
computed by the same predicate. `static/order-new.html` uses it to know whether
the order under construction contains anything physical.

When it contains nothing physical, the Pickup checkbox (`order-new.html:114`) is
**disabled** and reads *"No shipping — nothing physical in this order."* The
checkbox stops being something you must remember to tick and becomes a readout.

### 4. Per-client pickup default

New `dashboard/client_prefs.py`, shaped like its sibling `client_prices.py`:

- table `client_prefs`, keyed by normalized (lowercased, stripped) email
- one column, `pickup_default`, plus `updated_at`
- `get_pickup_default(cx, email) -> bool` — unknown email returns `False`
- `set_pickup_default(cx, email, value)` — idempotent

**Nothing writes it except an explicit operator action.** Saving an order does not
write it. `/api/orders/manual` does not write it. This is what keeps the biofield
hand-off invoices, with their deliberate `pickup: True`, from silently teaching
the system that every biofield client picks up.

In `static/order-new.html`, a second checkbox sits beside Pickup:

> **Pickup by default for this client**

It is visible only when a customer with an email is selected **and** the order
contains at least one shippable line. An order with nothing to ship represents no
fulfillment decision and cannot teach us one.

Ticking it POSTs immediately to a console-key-gated endpoint, on its own, not as
part of the order payload. It is a statement about the client, so it takes effect
without saving an order and is not undone by abandoning one.

**Applying it:**

- **New order** — selecting a customer reads the preference and pre-checks Pickup.
  Untick it and it stays unticked *for that order only*.
- **Edit** — the order's own `channel` wins, as it does today at
  `order-new.html:327`. An order saved as a pickup is a pickup, regardless of what
  the client's default has since become.

### 5. Biofield hand-off invoices keep free shipping — deliberately

`dashboard/biofield_invoice.py` continues to send `pickup: True`. This suppresses
shipping on the remedy bottles as well as the fee, and that is now an intentional
courtesy absorbed into the $300 analysis fee rather than an accident of the
`_price_cart` bug.

A comment at `biofield_invoice.py:119` records this, so it is not "fixed" later by
someone who reads this spec and assumes the flag was vestigial.

Existing orders are untouched. Nothing migrates. Orders already carrying
`channel='pickup'` keep it.

### Known issue, explicitly out of scope

Because hand-off invoices are modeled as pickups, the orders board labels them
"· Pickup" and `combined_shipments` treats them as "pickup (no shipment)" while
Rae in fact mails them. Expressing "free shipping" without claiming "pickup" would
need an order-level courtesy flag distinct from `channel`. That is a separate
change and is not part of this work.

## Testing

**The predicate.** `is_shippable` is false for `biofield-analysis`,
`evox-session`, and `emf`; true for an ordinary Functional Formulation.

**The pricing bug, pinned.** A cart of four bottles plus a Biofield Analysis must
be priced through the box-fit catalog on the bottle alone. Comparing invoice
totals is NOT sufficient: at today's rates the buggy path (qty fallback, 5 items
→ M) and the correct path (box-fit, 4 bottles → M) both return $23.00. The test
therefore spies on the `box_counts` handed to `dashboard.shipping.quote()` and
asserts the service does not appear in it. Verified 2026-07-08 by removing the
`is_shippable` gate and watching this test go red.

**Derived zero.** A services-only cart prices to `shipping_cents == 0` with Pickup
unticked and no flag set.

**Country.** An overseas services-only cart prices without raising `CheckoutError`.
An overseas cart containing a bottle still raises.

**The preference.** Default false; set/get round-trip; re-set is idempotent; email
is normalized. And the test that guards the whole design: **saving an order with
Pickup ticked leaves `client_prefs` empty.**

**Existing behavior stays pinned.** `tests/test_orders_effective_shipping.py`
unchanged. `tests/test_invoice_edit.py` pickup expectations unchanged.
`biofield_invoice` still sends `pickup: True`.

## Files touched

| File | Change |
|---|---|
| `dashboard/shipping.py` | add `is_shippable(product)` |
| `app.py` | `_price_cart`: skip non-shippable lines in `box_counts`/`total_bottles`; gate the non-US country check on having a shippable line |
| `dashboard/products.py` | emit `shippable` per product |
| `dashboard/client_prefs.py` | **new** — `client_prefs` table, get/set `pickup_default` |
| `app.py` | **new** console-gated GET/POST endpoints for the pickup default |
| `static/order-new.html` | disable Pickup when nothing is shippable; add the per-client default checkbox; pre-check Pickup from the preference on new orders |
| `dashboard/biofield_invoice.py` | comment only — `pickup: True` is a deliberate courtesy |
