from dashboard.orders import effective_shipping_cents

def test_pickup_zeroes_shipping():
    assert effective_shipping_cents(True, 1299) == 0
    assert effective_shipping_cents(True, 0) == 0
    assert effective_shipping_cents("pickup", 999) == 0   # any truthy

def test_non_pickup_passes_through():
    assert effective_shipping_cents(False, 1299) == 1299
    assert effective_shipping_cents(False, None) == 0
    assert effective_shipping_cents(False, "0") == 0
