import app

def test_helper_flags_device_bundle():
    # dental-bundle is a real device bundle (autoship_eligible False) after Task 2
    assert app._cart_has_noautoship_bundle([{"slug": "dental-bundle", "qty": 1}]) is True

def test_helper_allows_remedy_bundle():
    assert app._cart_has_noautoship_bundle([{"slug": "crystalline-lens-program", "qty": 1}]) is False

def test_helper_allows_single_sku():
    assert app._cart_has_noautoship_bundle([{"slug": "wholomega", "qty": 1}]) is False
