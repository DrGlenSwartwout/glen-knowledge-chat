# Pricing engine (dashboard/pricing.py)
Single source of truth for cart pricing. The one % discount per line = max(volume, subscriber
tier or coupon) — best-of-one; subscriber tier never stacks with a coupon. Volume is a smooth
months-based curve (`volume_anchors`, mix-and-match across all Functional Formulations;
Pure Powders and `info_only` excluded via `volume_eligible=false`; 30-cap bottle = 1 month,
larger formats via `months_per_unit`). Base is the true single-unit list, so floors anchor to
list: discount floor 57% (wholesale), points floor 43%. Override globally in
`pricing-settings.json` (DATA_DIR); override a single SKU with `sku_discount_floor_pct` /
`sku_points_floor_pct` or absolute `wholesale_cents` (Pure Powders use 0.75 → ~$30 at the
current ~$40 list; use absolute `wholesale_cents` if you want a fixed $30 regardless of list).
Points
(dashboard/points.py) earn 5% of full-price spend only, redeem above the floor, reduce the GET
tax base. Shipping is added by the caller at checkout (always charged, actual USA cost, US
ship-to only), never by the engine. Preview: POST /api/pricing/preview.
