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
