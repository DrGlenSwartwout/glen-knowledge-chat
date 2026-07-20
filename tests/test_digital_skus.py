"""Digital goods must be purchasable but never packed or charged postage.

Glen 2026-07-19 chose separate digital SKUs over collapsing the ebooks into the
print entries. Collapsing would have been wrong twice: the ebook is $9.97 vs
$19.97 print, and the print entry is `bottle_type: book` (shipped), so an ebook
buyer would have paid postage on a download.
"""
import app
from dashboard import shipping

EBOOKS = ("book-dry-eye-relief-ebook", "book-macular-regeneration-ebook")


def test_digital_flag_excludes_from_shipping():
    assert shipping.is_shippable({"digital": True}) is False
    assert shipping.is_shippable({"bottle_type": "book"}) is True   # print still ships
    assert shipping.is_shippable({}) is True                        # default loud


def test_ebooks_are_digital_purchasable_and_cheaper_than_print():
    for slug in EBOOKS:
        p = app._get_product(slug)
        assert p, slug
        assert p.get("digital") is True, f"{slug} must be flagged digital"
        assert not p.get("info_only"), f"{slug} must stay purchasable"
        assert not p.get("service"), f"{slug} is a good, not a consultation"
        assert p["price_cents"] == 997, slug
        assert shipping.is_shippable(p) is False, f"{slug} must not be packed"


def test_print_editions_still_ship_and_keep_their_own_price():
    for slug in ("book-dry-eye-relief", "book-macular-regeneration"):
        p = app._get_product(slug)
        assert p["price_cents"] == 1997
        assert p.get("bottle_type") == "book"
        assert shipping.is_shippable(p) is True


def test_no_digital_product_carries_a_bottle_type():
    """A digital SKU with a bottle_type is a contradiction that would confuse
    the packer if the digital flag were ever dropped."""
    for slug, p in app._PRODUCTS["products"].items():
        if p.get("digital"):
            assert not p.get("bottle_type"), slug


def test_vendor_shipped_devices_are_not_packed():
    """Glen 2026-07-20: Kloud mats ship from Centropix, NES miHealth from NES.

    We never pack these, so they must not enter the box packer. This cannot be
    expressed as flat_shipping_cents: 0 — app.py tests `_flat > 0`, so a zero
    falls through to the packer and an own-box device that quote() can't fit
    drops the cart to the coarse qty rule.
    """
    for slug in ("kloud-pemf-mini", "kloud-pemf-maxi", "nes-mihealth"):
        p = app._get_product(slug)
        assert p, slug
        assert p.get("vendor_shipped") is True, slug
        assert shipping.is_shippable(p) is False, f"{slug} must not be packed"
        assert not p.get("inactive"), f"{slug} must stay sellable"


def test_molecular_hydrogen_tablets_are_discontinued():
    """Discontinued 2026-07-20 — the ionizers produce molecular-hydrogen water."""
    assert app._get_product("molecular-hydrogen-tablets") is None  # inactive
    prompt = app.get_system_prompt("self-healing")
    assert "begin/product/molecular-hydrogen-tablets" not in prompt, \
        "prompt must not link a discontinued product"
    assert "Water Ionizer" in prompt


def test_portable_hydrogen_bottle_still_sellable():
    p = app._get_product("molecular-hydrogen-bottle")
    assert p and not p.get("inactive") and p["price_cents"] == 24997


def test_hydrogen_bottle_ships_large_box_not_ionizer_rate():
    """Glen 2026-07-20, correcting an earlier answer: the $100/unit rate belongs
    to the WATER IONIZERS. The Molecular Hydrogen bottle "would be a large box
    (now $32)".

    $32 is the USPS Large Flat Rate box under Glen's rounding rule ($31.50
    retail, >=50c rounds up). It is ours to ship, so not vendor_shipped.
    """
    p = app._get_product("molecular-hydrogen-bottle")
    assert p["flat_shipping_cents"] == 3200, "large box, not the ionizer rate"
    assert not p.get("vendor_shipped"), "we ship this, not the vendor"
    assert shipping.is_shippable(p) is True
    # the ionizers keep their own, higher rate
    assert app._get_product("water-ionizer-5plate")["flat_shipping_cents"] == 10000
