"""Services and digital products are not physical goods: they must never
contribute a bottle to the packer.

Before this, `resolve_bottle_type` fell back to the literal string "default"
for any product without a bottle mapping — including the `biofield-analysis`
service — so a cart of [analysis fee + 2 bottles] pushed an unknown type into
`shipping.quote()`, which raised `UnknownBottleType('default')`, silently fell
back to the coarse qty rule, and charged the customer for a phantom bottle.
"""
from dashboard.shipping import is_shippable


def test_service_is_not_shippable():
    assert is_shippable({"name": "Biofield Analysis", "service": True, "info_only": True}) is False
    assert is_shippable({"name": "EVOX Session", "service": True, "info_only": True}) is False


def test_digital_info_only_is_not_shippable():
    assert is_shippable({"name": "EMF", "info_only": True}) is False


def test_physical_product_is_shippable():
    assert is_shippable({"name": "30 ml dropper", "bottle_type": "30ml"}) is True
    assert is_shippable({"name": "Some Capsule"}) is True


def test_missing_or_falsy_flags_default_to_shippable():
    """A physical product with a missing bottle mapping must stay shippable so it
    still fails loudly rather than silently shipping for free."""
    assert is_shippable({}) is True
    assert is_shippable({"service": False, "info_only": False}) is True
    assert is_shippable(None) is True
