from dashboard.orders import channel_on_edit, effective_shipping_cents


def test_edit_sets_pickup_when_checked():
    assert channel_on_edit(True, None) == "pickup"
    assert channel_on_edit(True, "retail") == "pickup"
    assert channel_on_edit(True, "wholesale") == "pickup"


def test_edit_clears_pickup_when_unchecked():
    """Unchecking the box on a pickup order must return it to retail so shipping
    is charged again. Previously the channel latched: `pickup if pickup else
    (existing or "retail")` fell back to the stored "pickup" forever."""
    assert channel_on_edit(False, "pickup") == "retail"


def test_edit_unchecked_preserves_non_pickup_channel():
    """Unchecking must not flatten an order that was never a pickup — a wholesale
    order stays wholesale."""
    assert channel_on_edit(False, "wholesale") == "wholesale"
    assert channel_on_edit(False, "retail") == "retail"
    assert channel_on_edit(False, None) == "retail"
    assert channel_on_edit(False, "") == "retail"


def test_pickup_zeroes_shipping():
    assert effective_shipping_cents(True, 1299) == 0
    assert effective_shipping_cents(True, 0) == 0
    assert effective_shipping_cents("pickup", 999) == 0   # any truthy

def test_non_pickup_passes_through():
    assert effective_shipping_cents(False, 1299) == 1299
    assert effective_shipping_cents(False, None) == 0
    assert effective_shipping_cents(False, "0") == 0
