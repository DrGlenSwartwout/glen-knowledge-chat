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
