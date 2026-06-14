# Checkout on the pricing engine
`/reorder/checkout` prices via `dashboard.pricing.compute()` ONLY when
`PRICING_ENGINE_CHECKOUT` is truthy (default off → legacy `_qty_unit_cents` path).
When on: list-price QBO lines + a fixed-amount discount line (engine discount + redeemed
points) + a USPS shipping line; GET stays recorded-not-charged; ship-to must be US.
Orders record discount_cents / points_redeemed_cents / shipping_cents.
To go live: set PRICING_ENGINE_CHECKOUT=true in Doppler remedy-match/prd + Render.
Begin-funnel checkout, points earning at payment-return, and the Products-console floor
UI are later plans.
