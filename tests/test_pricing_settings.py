# tests/test_pricing_settings.py
from dashboard import pricing_settings as ps
from dashboard import pricing as _pricing
from dashboard import rewards as _rewards


def test_defaults_view_combines_pricing_and_rewards():
    d = ps.defaults_view()
    assert d["discount_floor_pct"] == _pricing.DEFAULTS["discount_floor_pct"]
    assert d["volume_anchors"] == _pricing.DEFAULTS["volume_anchors"]
    assert d["rewards"]["referral_reward_pct"] == _rewards.DEFAULTS["referral_reward_pct"]
    assert d["rewards"]["cash_out_threshold_cents"] == _rewards.DEFAULTS["cash_out_threshold_cents"]


def test_effective_empty_raw_equals_defaults():
    eff = ps.effective({})
    assert eff["discount_floor_pct"] == _pricing.DEFAULTS["discount_floor_pct"]
    assert eff["points_earn_pct"] == _pricing.DEFAULTS["points_earn_pct"]
    assert eff["rewards"]["cash_out_face_pct"] == _rewards.DEFAULTS["cash_out_face_pct"]
    assert "rewards" in eff and isinstance(eff["rewards"], dict)


def test_effective_overrides_merge():
    eff = ps.effective({"discount_floor_pct": 0.50, "rewards": {"referral_reward_pct": 0.08}})
    assert eff["discount_floor_pct"] == 0.50
    assert eff["points_floor_pct"] == _pricing.DEFAULTS["points_floor_pct"]
    assert eff["rewards"]["referral_reward_pct"] == 0.08
    assert eff["rewards"]["cash_out_threshold_cents"] == _rewards.DEFAULTS["cash_out_threshold_cents"]


def test_validate_accepts_full_valid_payload():
    payload = {
        "discount_floor_pct": 0.57, "points_floor_pct": 0.43, "points_earn_pct": 0.05,
        "points_redeem_per_point_cents": 5, "subscribe_tiers": [5, 10, 15],
        "cadences": [1, 2, 3], "volume_anchors": [[1, 0], [3, 14], [6, 29], [12, 43]],
        "rewards": {"referral_reward_pct": 0.05, "cash_out_threshold_cents": 10000,
                    "cash_out_face_pct": 0.70},
    }
    clean, errors = ps.validate(payload)
    assert errors == []
    assert clean["discount_floor_pct"] == 0.57
    assert clean["volume_anchors"] == [[1, 0], [3, 14], [6, 29], [12, 43]]
    assert clean["rewards"]["cash_out_threshold_cents"] == 10000


def test_validate_rejects_out_of_range_fractions():
    _, errors = ps.validate({"discount_floor_pct": 1.5})
    assert any("discount_floor_pct" in e for e in errors)
    _, errors = ps.validate({"points_earn_pct": -0.1})
    assert any("points_earn_pct" in e for e in errors)


def test_validate_rejects_points_floor_above_discount_floor():
    _, errors = ps.validate({"discount_floor_pct": 0.40, "points_floor_pct": 0.50})
    assert any("points_floor_pct" in e for e in errors)


def test_validate_rejects_bad_volume_anchors():
    _, errors = ps.validate({"volume_anchors": [[3, 14], [1, 0]]})
    assert any("volume_anchors" in e for e in errors)
    _, errors = ps.validate({"volume_anchors": [[1, 0], [3, 150]]})
    assert any("volume_anchors" in e for e in errors)
    _, errors = ps.validate({"volume_anchors": [[1, 0, 9]]})
    assert any("volume_anchors" in e for e in errors)


def test_validate_rejects_bad_rewards():
    _, errors = ps.validate({"rewards": {"cash_out_threshold_cents": -5}})
    assert any("cash_out_threshold_cents" in e for e in errors)
    _, errors = ps.validate({"rewards": {"cash_out_face_pct": 2.0}})
    assert any("cash_out_face_pct" in e for e in errors)


def test_validate_ignores_unknown_keys():
    clean, errors = ps.validate({"discount_floor_pct": 0.57, "bogus": 123})
    assert errors == []
    assert "bogus" not in clean


def test_validate_partial_payload_only_validates_present_keys():
    clean, errors = ps.validate({"points_earn_pct": 0.06})
    assert errors == []
    assert clean == {"points_earn_pct": 0.06}


def test_defaults_view_includes_referral_cert_anchors():
    d = ps.defaults_view()
    assert d["rewards"]["referral_cert_anchors"] == [[0, 5], [6, 10], [12, 15]]
    # must be a copy, not the shared module default
    d["rewards"]["referral_cert_anchors"][0][1] = 99
    assert _rewards.DEFAULTS["referral_cert_anchors"][0][1] == 5


def test_validate_accepts_referral_cert_anchors():
    clean, errors = ps.validate({"rewards": {"referral_cert_anchors": [[0, 5], [4, 9], [12, 15]]}})
    assert errors == []
    assert clean["rewards"]["referral_cert_anchors"] == [[0, 5], [4, 9], [12, 15]]


def test_validate_rejects_bad_referral_cert_anchors():
    _, e = ps.validate({"rewards": {"referral_cert_anchors": [[4, 9], [0, 5]]}})   # not ascending
    assert any("referral_cert_anchors" in x for x in e)
    _, e = ps.validate({"rewards": {"referral_cert_anchors": [[0, 150]]}})         # pct > 100
    assert any("referral_cert_anchors" in x for x in e)
    _, e = ps.validate({"rewards": {"referral_cert_anchors": [[-1, 5]]}})          # modules < 0
    assert any("referral_cert_anchors" in x for x in e)


def test_volume_anchors_still_validated_after_refactor():
    _, e = ps.validate({"volume_anchors": [[3, 14], [1, 0]]})   # not ascending
    assert any("volume_anchors" in x for x in e)
    clean, e2 = ps.validate({"volume_anchors": [[1, 0], [3, 14]]})
    assert e2 == [] and clean["volume_anchors"] == [[1, 0], [3, 14]]


def test_defaults_view_includes_discounts_block():
    d = ps.defaults_view()
    assert d["discounts"] == {
        "same_sku":      {"enabled": True,  "anchors": [[1, 0], [12, 29]]},
        "program_total": {"enabled": True,  "anchors": [[1, 0], [18, 29]]},
        "open_total":    {"enabled": False, "anchors": [[1, 0], [12, 0]]},
    }


def test_defaults_view_discounts_is_a_deep_copy():
    d = ps.defaults_view()
    d["discounts"]["same_sku"]["anchors"][0][1] = 99
    d["discounts"]["same_sku"]["enabled"] = False
    assert _pricing.DEFAULTS["discounts"]["same_sku"]["anchors"][0][1] == 0
    assert _pricing.DEFAULTS["discounts"]["same_sku"]["enabled"] is True


def test_effective_empty_has_default_discounts():
    eff = ps.effective({})
    assert eff["discounts"] == _pricing.DEFAULTS["discounts"]
    assert eff["discounts"]["open_total"]["enabled"] is False


def test_effective_legacy_volume_anchors_backfills_open_total():
    eff = ps.effective({"volume_anchors": [[1, 0], [3, 15], [6, 30], [12, 45]]})
    assert eff["discounts"]["open_total"]["anchors"] == [[1, 0], [3, 15], [6, 30], [12, 45]]
    assert eff["discounts"]["open_total"]["enabled"] is False
    # other types still fall to code defaults
    assert eff["discounts"]["same_sku"] == {"enabled": True, "anchors": [[1, 0], [12, 29]]}
    assert eff["discounts"]["program_total"] == {"enabled": True, "anchors": [[1, 0], [18, 29]]}


def test_effective_explicit_discounts_block_passes_through():
    payload = {
        "discounts": {
            "same_sku": {"enabled": False, "anchors": [[1, 0], [10, 20]]},
            "program_total": {"enabled": True, "anchors": [[1, 0], [12, 29]]},
            "open_total": {"enabled": True, "anchors": [[1, 0], [12, 10]]},
        }
    }
    eff = ps.effective(payload)
    assert eff["discounts"] == payload["discounts"]


def test_validate_accepts_well_formed_discounts_payload():
    payload = {
        "discounts": {
            "same_sku": {"enabled": True, "anchors": [[1, 0], [12, 29]]},
            "program_total": {"enabled": False, "anchors": [[1, 0], [12, 29]]},
            "open_total": {"enabled": True, "anchors": [[1, 0], [6, 10], [12, 20]]},
        }
    }
    clean, errors = ps.validate(payload)
    assert errors == []
    assert clean["discounts"] == payload["discounts"]


def test_validate_rejects_non_bool_enabled():
    _, e = ps.validate({"discounts": {"same_sku": {"enabled": "yes", "anchors": [[1, 0], [12, 29]]}}})
    assert any("discounts.same_sku.enabled" in x for x in e)


def test_validate_rejects_non_ascending_discount_anchors():
    _, e = ps.validate({"discounts": {"same_sku": {"enabled": True, "anchors": [[12, 29], [1, 0]]}}})
    assert any("discounts.same_sku.anchors" in x for x in e)


def test_validate_rejects_discount_pct_over_100():
    _, e = ps.validate({"discounts": {"open_total": {"enabled": True, "anchors": [[1, 0], [12, 150]]}}})
    assert any("discounts.open_total.anchors" in x for x in e)


def test_validate_discounts_partial_types_only():
    clean, errors = ps.validate({"discounts": {"same_sku": {"enabled": True, "anchors": [[1, 0], [12, 29]]}}})
    assert errors == []
    assert clean["discounts"] == {"same_sku": {"enabled": True, "anchors": [[1, 0], [12, 29]]}}
