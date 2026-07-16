from dashboard.subscriptions import tier_for, tier_for_bundle, BUNDLE_SUBSCRIBE_TIERS


def test_bundle_ladder_values():
    assert BUNDLE_SUBSCRIBE_TIERS == [12, 14, 16, 18, 20, 22, 24, 26, 28, 29]


def test_tier_for_bundle_first_and_cap():
    assert tier_for_bundle(0) == 12
    assert tier_for_bundle(1) == 14
    assert tier_for_bundle(9) == 29
    assert tier_for_bundle(50) == 29   # clamped at cap


def test_single_ladder_unchanged():
    assert tier_for(0) == 3
    assert tier_for(11) == 25
    assert tier_for(99) == 25
