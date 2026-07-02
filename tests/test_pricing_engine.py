import pytest

from dashboard import pricing


def test_defaults_present():
    s = pricing.load_settings({})
    assert s["discount_floor_pct"] == 0.57
    assert s["points_floor_pct"] == 0.43
    assert s["points_earn_pct"] == 0.05
    assert s["points_redeem_per_point_cents"] == 5
    assert s["subscribe_tiers"] == [5, 10, 15]
    assert s["cadences"] == [1, 2, 3]
    assert s["volume_anchors"] == [[1, 0], [12, 29]]


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


def test_points_reduce_above_floor():
    # price 5950, points 1000, floor 3010 → 4950, used 1000
    assert pricing.apply_points(5950, 1000, 3010) == (4950, 1000)


def test_points_clamped_at_floor_partial_use():
    # price 4000, want 2000 off, floor 3010 → only 990 usable → 3010, used 990
    assert pricing.apply_points(4000, 2000, 3010) == (3010, 990)


def test_points_none_requested():
    assert pricing.apply_points(5950, 0, 3010) == (5950, 0)


def test_volume_pct_at_anchors():
    s = pricing.load_settings({})
    # LINEAR ramp: pct = 29*(q-1)/11, 0% at qty 1 → 29% at qty 12, flat beyond.
    assert pricing.volume_pct(1, s) == 0
    assert pricing.volume_pct(12, s) == 29
    assert pricing.volume_pct(2, s) == pytest.approx(29 * 1 / 11)
    assert pricing.volume_pct(4, s) == pytest.approx(29 * 3 / 11)
    assert pricing.volume_pct(7, s) == pytest.approx(29 * 6 / 11)
    assert pricing.volume_pct(3, s) == pytest.approx(29 * 2 / 11)
    assert pricing.volume_pct(6, s) == pytest.approx(29 * 5 / 11)


def test_volume_pct_interpolates_and_caps():
    s = pricing.load_settings({})
    assert pricing.volume_pct(5, s) == pytest.approx(29 * 4 / 11)
    assert pricing.volume_pct(24, s) == 29
    assert pricing.volume_pct(0, s) == 0


def test_volume_pct_unchanged_after_ramp_refactor():
    # Behavior-preservation check: same values as test_volume_pct_at_anchors/interpolates_and_caps.
    s = pricing.load_settings({})
    assert pricing.volume_pct(1, s) == 0
    assert pricing.volume_pct(12, s) == 29


def test_same_sku_pct_default_on_linear():
    s = pricing.load_settings({})
    assert pricing.same_sku_pct(1, s) == 0
    assert pricing.same_sku_pct(12, s) == 29
    assert pricing.same_sku_pct(6, s) == pytest.approx(29 * 5 / 11)
    assert pricing.same_sku_pct(99, s) == 29


def test_open_total_pct_default_off_then_on_via_override():
    s = pricing.load_settings({})
    assert pricing.open_total_pct(1, s) == 0
    assert pricing.open_total_pct(12, s) == 0
    assert pricing.open_total_pct(99, s) == 0

    s2 = pricing.load_settings({"discounts": {
        "same_sku":      {"enabled": True, "anchors": [[1, 0], [12, 29]]},
        "program_total": {"enabled": True, "anchors": [[1, 0], [12, 29]]},
        "open_total":    {"enabled": True, "anchors": [[1, 0], [12, 20]]},
    }})
    assert pricing.open_total_pct(12, s2) == 20


def test_program_total_pct_gated_on_membership():
    s = pricing.load_settings({})
    assert pricing.program_total_pct(12, s, program_member=False) == 0
    assert pricing.program_total_pct(12, s, program_member=True) == 29


def test_discount_cfg_back_compat_from_legacy_volume_anchors():
    # Note: pricing.load_settings({"volume_anchors": ...}) always carries DEFAULTS["discounts"]
    # (present+truthy), so this exercises _discount_cfg's legacy-fallback branch directly with a
    # settings dict that predates the "discounts" key (e.g. an old on-disk pricing-settings.json).
    s = {"volume_anchors": [[1, 0], [12, 40]]}
    cfg = pricing._discount_cfg(s)
    assert cfg["open_total"] == {"enabled": False, "anchors": [[1, 0], [12, 40]]}
    assert cfg["same_sku"]["enabled"] is True


def _fake_tax(subtotal_cents, *, channel, ship_to_state, resale_ok=False):
    return int(round(subtotal_cents * 0.04)) if ship_to_state == "HI" else 0


def test_compute_one_line_subscriber_tier_and_points():
    s = pricing.load_settings({})
    items = [{"slug": "neuro-mag", "name": "Neuro Mag", "qty": 1,
              "product": {"slug": "neuro-mag", "price_cents": 7000},
              "unit_cents": 7000, "months": 1, "volume_eligible": True}]
    r = pricing.compute(items, settings=s, subscriber_tier_pct=15,
                        points_to_redeem_cents=1000, channel="retail",
                        ship_to_state="HI", tax_fn=_fake_tax)
    # M=1 -> volume 0%; best-of(0,15)=15%. 15% off 7000 = 5950; points 1000 -> 4950
    assert r["lines"][0]["line_total_cents"] == 4950
    assert r["discount_cents"] == 1050
    assert r["points_redeemed_cents"] == 1000
    assert r["get_cents"] == 198            # round(4950*0.04)


def test_compute_subscriber_tier_beats_coupon_no_stack():
    s = pricing.load_settings({})
    items = [{"slug": "x", "name": "X", "qty": 1,
              "product": {"slug": "x", "price_cents": 7000},
              "unit_cents": 7000, "months": 1, "volume_eligible": True}]
    r = pricing.compute(items, settings=s, subscriber_tier_pct=5, coupon_pct=40,
                        channel="retail", ship_to_state="HI", tax_fn=_fake_tax)
    assert r["lines"][0]["line_total_cents"] == 6650   # 5% (sub), coupon ignored


def test_compute_volume_mix_and_match_beats_subscriber():
    s = pricing.load_settings({})
    # two different SKUs, 6 months each = 12 total -> linear volume 29% beats 15% tier
    items = [
        {"slug": "a", "name": "A", "qty": 1, "product": {"slug": "a", "price_cents": 7000},
         "unit_cents": 7000, "months": 6, "volume_eligible": True},
        {"slug": "b", "name": "B", "qty": 1, "product": {"slug": "b", "price_cents": 7000},
         "unit_cents": 7000, "months": 6, "volume_eligible": True},
    ]
    r = pricing.compute(items, settings=s, subscriber_tier_pct=15,
                        channel="retail", ship_to_state="CA", tax_fn=_fake_tax)
    assert r["lines"][0]["line_total_cents"] == 4970   # round(7000*(1-0.29))
    assert r["lines"][1]["line_total_cents"] == 4970


def test_compute_pure_powder_excluded_from_volume_floored_at_30():
    s = pricing.load_settings({})
    # Pure Powder: NOT volume_eligible (months ignored), per-SKU floor 75% of 4000 = 3000
    items = [{"slug": "pp", "name": "Pure Powder", "qty": 1,
              "product": {"slug": "pp", "price_cents": 4000,
                          "sku_discount_floor_pct": 0.75, "sku_points_floor_pct": 0.75},
              "unit_cents": 4000, "months": 12, "volume_eligible": False}]
    r = pricing.compute(items, settings=s, subscriber_tier_pct=15,
                        points_to_redeem_cents=1000, channel="retail",
                        ship_to_state="CA", tax_fn=_fake_tax)
    # volume 0 (excluded); 15% off 4000 = 3400 (floor 3000); points -> 3000 (only 400 used)
    assert r["lines"][0]["line_total_cents"] == 3000
    assert r["points_redeemed_cents"] == 400


def test_compute_empty_cart():
    s = pricing.load_settings({})
    r = pricing.compute([], settings=s, tax_fn=_fake_tax)
    assert r["subtotal_cents"] == 0
    assert r["lines"] == []
    assert r["total_cents"] == 0


def test_compute_zero_qty_line():
    s = pricing.load_settings({})
    items = [{"slug": "neuro-mag", "name": "Neuro Mag", "qty": 0,
              "product": {"slug": "neuro-mag", "price_cents": 7000},
              "unit_cents": 7000, "months": 1, "volume_eligible": True}]
    r = pricing.compute(items, settings=s, tax_fn=_fake_tax)
    assert r["lines"][0]["line_total_cents"] == 0
    assert r["subtotal_cents"] == 0


def test_compute_points_exceed_all_floor_headroom():
    s = pricing.load_settings({})
    # Two items, each 7000 list, no other discount applied (0%).
    # Discount floor = round(7000*0.57) = 3990; points floor = round(7000*0.43) = 3010.
    # After 0% discount, price stays at 7000 per line.
    # Reducible headroom per line = 7000 - 3010 = 3990 each; total = 7980.
    # We request way more than that (20000) → should clamp and only consume 7980.
    items = [
        {"slug": "a", "name": "A", "qty": 1,
         "product": {"slug": "a", "price_cents": 7000},
         "unit_cents": 7000, "months": 1, "volume_eligible": False},
        {"slug": "b", "name": "B", "qty": 1,
         "product": {"slug": "b", "price_cents": 7000},
         "unit_cents": 7000, "months": 1, "volume_eligible": False},
    ]
    r = pricing.compute(items, settings=s, points_to_redeem_cents=20000,
                        tax_fn=_fake_tax)
    assert r["lines"][0]["line_total_cents"] == 3010   # clamped at points floor
    assert r["lines"][1]["line_total_cents"] == 3010   # clamped at points floor
    assert r["points_redeemed_cents"] == 7980          # total headroom consumed, not 20000


def test_unit_floor_unknown_kind_raises():
    s = pricing.load_settings({})
    try:
        pricing.unit_floor_cents({"price_cents": 7000}, 7000, s, "bogus")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "bogus" in str(e)


def test_floor_honors_zero_wholesale():
    s = pricing.load_settings({})
    # wholesale_cents=0 is an explicit override; discount floor must be 0, not fall through to pct
    p = {"slug": "free", "price_cents": 7000, "wholesale_cents": 0}
    assert pricing.unit_floor_cents(p, 7000, s, "discount") == 0


def test_curve_lands_target_cents_on_6997_list():
    s = pricing.load_settings({})
    L = 6997
    expected = {1: 6997, 2: 6813, 4: 6444, 7: 5890, 12: 4968, 99: 4968}
    for m, cents in expected.items():
        pct = pricing.volume_pct(m, s)
        assert int(round(L * (1 - pct / 100.0))) == cents
    assert pricing.unit_floor_cents({"slug": "ff", "price_cents": L}, L, s, "discount") == 3988
