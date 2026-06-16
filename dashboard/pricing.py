"""Single source of truth for cart pricing: one % discount, points, and the
wholesale (57%) / points (43%) floors. Pure + injectable for testing."""

DEFAULTS = {
    "discount_floor_pct": 0.57,           # all % discounts clamp up to list * this (= wholesale)
    "points_floor_pct": 0.43,             # points clamp up to list * this
    "points_earn_pct": 0.05,              # earn 5% of full-price spend, as redemption-value cents
    "points_redeem_per_point_cents": 5,   # 1 point = 5 cents (20 points = $1)
    "subscribe_tiers": [5, 10, 15],       # % by completed-order count (1st,2nd,3rd+)
    "cadences": [1, 2, 3],                # months
    # volume curve: [total_months, pct_off] knots, ascending; linear interp; flat beyond last.
    # A smooth LINEAR ramp from 1 unit (0%) to 12 units (max), flat at the max beyond 12.
    # Two knots = one straight line; edit the last row's % in the console to set the max.
    "volume_anchors": [[1, 0], [12, 43]],
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
        if whole is not None:
            return int(whole)
        pct = product.get("sku_discount_floor_pct", settings["discount_floor_pct"])
        return int(round(list_cents * pct))
    if kind == "points":
        if whole is not None:
            allowance = int(round(list_cents * (settings["discount_floor_pct"]
                                                - settings["points_floor_pct"])))
            return int(whole) - allowance
        pct = product.get("sku_points_floor_pct", settings["points_floor_pct"])
        return int(round(list_cents * pct))
    raise ValueError(f"unknown kind: {kind!r}")


def apply_discount(list_cents, pct, floor_cents):
    """Apply a single percentage discount, never below floor_cents."""
    discounted = int(round(int(list_cents) * (1 - (pct or 0) / 100.0)))
    return max(discounted, int(floor_cents))


def apply_points(price_cents, points_cents, floor_cents):
    """Subtract points (in redemption-value cents) but never below floor_cents.
    Returns (new_price_cents, points_actually_used_cents)."""
    price_cents = int(price_cents)
    reducible = max(0, price_cents - int(floor_cents))
    used = min(max(0, int(points_cents)), reducible)
    return price_cents - used, used


def volume_pct(months, settings):
    """Percentage discount for total cart months, linear-interpolated through the
    console anchor table (ascending [months, pct_off] pairs); flat beyond the last knot."""
    anchors = settings["volume_anchors"]
    m = max(0, int(months or 0))
    if m <= anchors[0][0]:
        return float(anchors[0][1])
    for (m0, p0), (m1, p1) in zip(anchors, anchors[1:]):
        if m <= m1:
            return p0 + (p1 - p0) * (m - m0) / (m1 - m0)
    return float(anchors[-1][1])


def compute(items, *, settings, subscriber_tier_pct=None, coupon_pct=None,
            points_to_redeem_cents=0, channel="retail", ship_to_state=None,
            resale_ok=False, tax_fn=None):
    """Price a cart. The single % discount per line = max(volume_pct, sub-or-coupon).
    Subscriber tier and coupon never stack (subscriber wins if present). Points apply
    after, then GET tax on the discounted subtotal. Base is the TRUE single-unit list, so
    floors always anchor to list.

    items: [{"slug","name","qty","product","unit_cents","months","volume_eligible"}]
    Returns a dict with per-line breakdown + order totals.

    Points are allocated greedily in item-list order; the first item consumes points first.
    """
    base_pct = coupon_pct or 0
    if subscriber_tier_pct is not None:
        base_pct = subscriber_tier_pct      # subscriber tier wins whenever present, even 0
    total_months = sum(int(it.get("months") or 0) for it in items if it.get("volume_eligible"))
    vpct = volume_pct(total_months, settings)
    points_left = max(0, int(points_to_redeem_cents or 0))
    lines, subtotal, total_discount, total_points = [], 0, 0, 0

    for it in items:
        p = it["product"]
        qty = int(it["qty"])
        unit_list = int(it["unit_cents"])
        line_list = unit_list * qty
        line_pct = max(vpct if it.get("volume_eligible") else 0, base_pct)  # best-of-one: volume only for eligible items; base discount always applies
        disc_floor = unit_floor_cents(p, unit_list, settings, "discount") * qty
        pts_floor = unit_floor_cents(p, unit_list, settings, "points") * qty

        after_disc = apply_discount(line_list, line_pct, disc_floor)
        after_pts, used = apply_points(after_disc, points_left, pts_floor)
        points_left -= used

        lines.append({
            "slug": it["slug"], "name": it["name"], "qty": qty,
            "list_cents": line_list, "discount_cents": line_list - after_disc,
            "points_cents": used, "line_total_cents": after_pts, "pct_applied": line_pct,
        })
        subtotal += after_pts
        total_discount += (line_list - after_disc)
        total_points += used

    get_cents = tax_fn(subtotal, channel=channel, ship_to_state=ship_to_state,
                       resale_ok=resale_ok) if tax_fn else 0
    return {
        "lines": lines,
        "subtotal_cents": subtotal,
        "discount_cents": total_discount,
        "points_redeemed_cents": total_points,
        "volume_months": total_months,
        "volume_pct": vpct,
        "get_cents": get_cents,
        "total_cents": subtotal + get_cents,
    }
