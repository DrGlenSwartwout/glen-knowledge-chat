import app

def _item(product):
    return {"slug": "x", "name": "x", "qty": 1, "product": product,
            "unit_cents": 10000, "months": 1, "volume_eligible": True}

def test_bundle_item_uses_bundle_ladder():
    r = app._subscription_tier_resolver(0, True)
    assert r(_item({"bundle": True, "autoship_eligible": True})) == 12

def test_single_item_uses_standard_ladder():
    r = app._subscription_tier_resolver(0, True)
    assert r(_item({})) == 3

def test_device_bundle_not_eligible_uses_standard_ladder():
    # a bundle flagged autoship_eligible False should NOT get the bundle ladder
    r = app._subscription_tier_resolver(0, True)
    assert r(_item({"bundle": True, "autoship_eligible": False})) == 3

def test_inactive_membership_zeroes_all():
    r = app._subscription_tier_resolver(5, False)
    assert r(_item({"bundle": True, "autoship_eligible": True})) == 0
    assert r(_item({})) == 0

def test_climbs_with_order_count():
    r = app._subscription_tier_resolver(9, True)
    assert r(_item({"bundle": True, "autoship_eligible": True})) == 29  # bundle cap
    assert r(_item({})) == 21                                           # tier_for(9)
