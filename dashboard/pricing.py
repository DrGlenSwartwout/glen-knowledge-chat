"""Single source of truth for cart pricing: one % discount, points, and the
wholesale (57%) / points (43%) floors. Pure + injectable for testing."""

DEFAULTS = {
    "discount_floor_pct": 0.57,           # all % discounts clamp up to list * this (= wholesale)
    "points_floor_pct": 0.43,             # points clamp up to list * this
    "points_earn_pct": 0.05,              # earn 5% of full-price spend, as redemption-value cents
    "points_redeem_per_point_cents": 5,   # 1 point = 5 cents (20 points = $1)
    "subscribe_tiers": [5, 10, 15],       # % by completed-order count (1st,2nd,3rd+)
    "cadences": [1, 2, 3],                # months
    # volume ramp: [total_months, pct_off] knots, ascending; linear interp; flat beyond last.
    # A straight LINEAR line from 1 unit (0% off, base $70) to 12 units (29% off, floor $50),
    # flat beyond 12 — two anchors only, so the per-unit discount grows evenly with quantity
    # (not steep-early). Edit rows in the console to reshape.
    "volume_anchors": [[1, 0], [12, 29]],
    "repertoire_reorder_pct": 0.29,   # member flat reorder rate on repertoire SKUs (~$50 on $69.97)
    # Absolute per-unit price floor for volume-eligible FF (qty_pricing, not info_only):
    # the ramp's 29% off $69.97 lands at $49.68, but the FF minimum unit price is a clean
    # $50. Applied only when list >= this floor, so cheaper FFs (e.g. 50%-off powders)
    # keep their own lower wholesale floor + full volume discount. Glen 2026-07-11.
    "ff_min_unit_cents": 5000,
    # discount TYPES (console-toggleable, non-additive; see pricing_settings.py):
    # same_sku = per-line SKU qty (open to everyone); program_total = order-total,
    # gated on paid membership; open_total = order-total, everyone (default OFF —
    # conflicts with the public store; inherits legacy volume_anchors on back-compat).
    "discounts": {
        "same_sku":      {"enabled": True,  "anchors": [[1, 0], [12, 29]]},
        "program_total": {"enabled": True,  "anchors": [[1, 0], [18, 29]]},
        "open_total":    {"enabled": False, "anchors": [[1, 0], [12, 0]]},
    },
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
            base_floor = int(whole)
        else:
            pct = product.get("sku_discount_floor_pct", settings["discount_floor_pct"])
            base_floor = int(round(list_cents * pct))
        # FF minimum unit price: a volume-eligible FF (qty_pricing, not info_only) never
        # discounts below ff_min_unit_cents ($50) — but only when its list is at/above that
        # floor, so a cheap FF keeps its own lower floor and full volume discount. Explicit
        # per-client/override prices bypass this (callers return the override before the floor).
        ff_min = int(settings.get("ff_min_unit_cents") or 0)
        if (ff_min and list_cents >= ff_min
                and product.get("qty_pricing") and not product.get("info_only")):
            return max(base_floor, ff_min)
        return base_floor
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


def _ramp_pct(qty, anchors):
    """Linear-interpolated pct through ascending [qty, pct] anchors; flat beyond the last."""
    q = max(0, int(qty or 0))
    if q <= anchors[0][0]:
        return float(anchors[0][1])
    for (m0, p0), (m1, p1) in zip(anchors, anchors[1:]):
        if q <= m1:
            return p0 + (p1 - p0) * (q - m0) / (m1 - m0)
    return float(anchors[-1][1])


def volume_pct(months, settings):
    """Legacy OWNER in-house / product-page order-total ramp (reads volume_anchors)."""
    return _ramp_pct(months, settings["volume_anchors"])


def _discount_cfg(settings):
    """Back-compat mirror of pricing_settings.effective(): return settings['discounts']
    if present+truthy, else synthesize from legacy volume_anchors with open_total DISABLED."""
    d = settings.get("discounts")
    if isinstance(d, dict) and d:
        return d
    legacy = settings.get("volume_anchors") or DEFAULTS["volume_anchors"]
    base = DEFAULTS["discounts"]
    return {
        "same_sku":      {"enabled": base["same_sku"]["enabled"],
                          "anchors": [list(a) for a in base["same_sku"]["anchors"]]},
        "program_total": {"enabled": base["program_total"]["enabled"],
                          "anchors": [list(a) for a in base["program_total"]["anchors"]]},
        "open_total":    {"enabled": False, "anchors": [list(a) for a in legacy]},
    }


def same_sku_pct(line_qty, settings):
    """Per-line SKU-quantity discount pct (open to everyone, default ON)."""
    d = _discount_cfg(settings)["same_sku"]
    return _ramp_pct(line_qty, d["anchors"]) if d.get("enabled") else 0.0


def program_total_pct(total_qty, settings, program_member):
    """Order-total discount pct, gated on paid-program membership (default ON)."""
    d = _discount_cfg(settings)["program_total"]
    return _ramp_pct(total_qty, d["anchors"]) if (d.get("enabled") and program_member) else 0.0


def open_total_pct(total_qty, settings):
    """Order-total discount pct, open to everyone (default OFF)."""
    d = _discount_cfg(settings)["open_total"]
    return _ramp_pct(total_qty, d["anchors"]) if d.get("enabled") else 0.0


def compute(items, *, settings, subscriber_tier_pct=None, coupon_pct=None,
            points_to_redeem_cents=0, channel="retail", ship_to_state=None,
            resale_ok=False, tax_fn=None, program_member=False, repertoire_slugs=None):
    """Price a cart. The single % discount per line = the best (non-additive) of:
    type1 same-SKU (this line's own qty, open to everyone, default ON), type2
    program-total (order-total months, gated on program_member, default ON) or
    type3 open-total (order-total months, everyone, default OFF), and sub-or-coupon.
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
    open_pct = open_total_pct(total_months, settings)
    prog_pct = program_total_pct(total_months, settings, program_member)
    points_left = max(0, int(points_to_redeem_cents or 0))
    lines, subtotal, total_discount, total_points = [], 0, 0, 0

    for it in items:
        p = it["product"]
        qty = int(it["qty"])
        unit_list = int(it["unit_cents"])
        line_list = unit_list * qty
        eligible = bool(it.get("volume_eligible"))
        t1 = same_sku_pct(qty, settings) if eligible else 0.0       # type1: this line's SKU qty
        order_pct = max(prog_pct, open_pct) if eligible else 0.0     # type2 (gated) / type3
        rep_pct = 0.0
        if repertoire_slugs and eligible and (it.get("slug") or "").strip().lower() in repertoire_slugs:
            # repertoire_reorder_pct is stored as a 0-1 fraction (like discount_floor_pct);
            # line_pct/apply_discount work in 0-100 percent, so convert here.
            rep_pct = float(settings.get("repertoire_reorder_pct") or 0.0) * 100.0
        line_pct = max(t1, order_pct, base_pct, rep_pct)             # non-additive: best single offer
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
        "volume_pct": max(prog_pct, open_pct),
        "get_cents": get_cents,
        "total_cents": subtotal + get_cents,
    }
