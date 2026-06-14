from dashboard import pricing


def test_defaults_present():
    s = pricing.load_settings({})
    assert s["discount_floor_pct"] == 0.57
    assert s["points_floor_pct"] == 0.43
    assert s["points_earn_pct"] == 0.05
    assert s["points_redeem_per_point_cents"] == 5
    assert s["subscribe_tiers"] == [5, 10, 15]
    assert s["cadences"] == [1, 2, 3]
    assert s["volume_anchors"] == [[1, 0], [3, 14], [6, 29], [12, 43]]


def test_overrides_merge_over_defaults():
    s = pricing.load_settings({"discount_floor_pct": 0.70})
    assert s["discount_floor_pct"] == 0.70   # overridden
    assert s["points_floor_pct"] == 0.43     # default retained


def test_floor_uses_global_pct_when_no_override():
    s = pricing.load_settings({})
    p = {"slug": "neuro-mag", "price_cents": 7000}
    assert pricing.unit_floor_cents(p, 7000, s, "discount") == 3990   # round(7000*0.57)
    assert pricing.unit_floor_cents(p, 7000, s, "points") == 3010     # round(7000*0.43)


def test_floor_uses_per_sku_pct_override():
    s = pricing.load_settings({})
    p = {"slug": "costly", "price_cents": 9000,
         "sku_discount_floor_pct": 0.70, "sku_points_floor_pct": 0.60}
    assert pricing.unit_floor_cents(p, 9000, s, "discount") == 6300   # 9000*0.70
    assert pricing.unit_floor_cents(p, 9000, s, "points") == 5400     # 9000*0.60


def test_floor_uses_absolute_wholesale_override():
    s = pricing.load_settings({})
    # absolute wholesale wins for the discount floor; points floor = wholesale - allowance,
    # allowance defaults to list*(discount_pct - points_pct) = 7000*0.14 = 980
    p = {"slug": "fixed", "price_cents": 7000, "wholesale_cents": 4200}
    assert pricing.unit_floor_cents(p, 7000, s, "discount") == 4200
    assert pricing.unit_floor_cents(p, 7000, s, "points") == 3220     # 4200 - 980


def test_discount_applied_above_floor():
    # 15% off 7000 = 5950, above the 3990 floor → 5950
    assert pricing.apply_discount(7000, 15, 3990) == 5950


def test_discount_clamped_to_floor():
    # 50% off 7000 = 3500, below the 3990 floor → clamp to 3990
    assert pricing.apply_discount(7000, 50, 3990) == 3990


def test_zero_discount_is_list():
    assert pricing.apply_discount(7000, 0, 3990) == 7000
