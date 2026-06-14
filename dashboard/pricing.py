"""Single source of truth for cart pricing: one % discount, points, and the
wholesale (57%) / points (43%) floors. Pure + injectable for testing."""

DEFAULTS = {
    "discount_floor_pct": 0.57,           # all % discounts clamp up to list * this (= wholesale)
    "points_floor_pct": 0.43,             # points clamp up to list * this
    "points_earn_pct": 0.05,              # earn 5% of full-price spend, as redemption-value cents
    "points_redeem_per_point_cents": 5,   # 1 point = 5 cents (20 points = $1)
    "subscribe_tiers": [5, 10, 15],       # % by completed-order count (1st,2nd,3rd+)
    "cadences": [1, 2, 3],                # months
    # volume curve: [total_months, pct_off] knots, ascending; linear interp; flat beyond last
    "volume_anchors": [[1, 0], [3, 14], [6, 29], [12, 43]],
}


def load_settings(overrides):
    """DEFAULTS merged with a dict of overrides (e.g. from pricing-settings.json)."""
    s = dict(DEFAULTS)
    for k, v in (overrides or {}).items():
        if v is not None:
            s[k] = v
    return s


def unit_floor_cents(product, list_cents, settings, kind):
    """Per-unit floor in cents. kind in ('discount','points').
    Precedence: absolute wholesale_cents > per-SKU pct > global pct."""
    list_cents = int(list_cents)
    whole = product.get("wholesale_cents")
    if kind == "discount":
        if whole:
            return int(whole)
        pct = product.get("sku_discount_floor_pct", settings["discount_floor_pct"])
        return int(round(list_cents * pct))
    # points
    if whole:
        allowance = int(round(list_cents * (settings["discount_floor_pct"]
                                            - settings["points_floor_pct"])))
        return int(whole) - allowance
    pct = product.get("sku_points_floor_pct", settings["points_floor_pct"])
    return int(round(list_cents * pct))
