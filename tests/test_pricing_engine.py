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
