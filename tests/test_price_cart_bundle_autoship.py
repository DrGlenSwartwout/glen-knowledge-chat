# tests/test_price_cart_bundle_autoship.py
"""End-to-end: _price_cart routes the per-line bundle autoship ladder (12->29)
to real pricing.compute against the real catalog. A bundle line earns the bundle
ladder; a single SKU in the same cart earns the standard ladder; inactive
membership zeroes both."""
import pytest

# app import needs the repo env (Pinecone/Doppler); skip cleanly if unavailable
# so bare `pytest` doesn't hard-error. Run this file under:
#   doppler run --project remedy-match --config dev -- python3 -m pytest tests/test_price_cart_bundle_autoship.py -q
app = pytest.importorskip("app")

SHIP = {"country": "US", "state": "HI", "name": "Test"}


def _by_slug(pc):
    return {ln["slug"]: ln for ln in pc["priced"]["lines"]}


def test_bundle_line_gets_bundle_ladder_single_gets_standard():
    # crystalline-lens-program is an autoship-eligible bundle; wholomega is a single SKU.
    cart = [{"slug": "crystalline-lens-program", "qty": 1},
            {"slug": "wholomega", "qty": 1}]
    pc = app._price_cart(cart, ship=SHIP, subscriber_order_count=0, subscriber_active=True)
    lines = _by_slug(pc)
    assert lines["crystalline-lens-program"]["pct_applied"] == 12  # tier_for_bundle(0)
    assert lines["wholomega"]["pct_applied"] == 3                  # tier_for(0)


def test_bundle_ladder_climbs_with_order_count():
    cart = [{"slug": "crystalline-lens-program", "qty": 1}]
    pc = app._price_cart(cart, ship=SHIP, subscriber_order_count=9, subscriber_active=True)
    assert _by_slug(pc)["crystalline-lens-program"]["pct_applied"] == 29  # bundle cap


def test_device_bundle_line_gets_standard_ladder_not_bundle():
    # dental-bundle is a bundle but autoship_eligible False -> standard ladder if ever priced.
    cart = [{"slug": "dental-bundle", "qty": 1}]
    pc = app._price_cart(cart, ship=SHIP, subscriber_order_count=0, subscriber_active=True)
    assert _by_slug(pc)["dental-bundle"]["pct_applied"] == 3  # tier_for(0), NOT 12


def test_inactive_membership_zeroes_all_lines():
    cart = [{"slug": "crystalline-lens-program", "qty": 1},
            {"slug": "wholomega", "qty": 1}]
    pc = app._price_cart(cart, ship=SHIP, subscriber_order_count=5, subscriber_active=False)
    lines = _by_slug(pc)
    assert lines["crystalline-lens-program"]["pct_applied"] == 0
    assert lines["wholomega"]["pct_applied"] == 0
