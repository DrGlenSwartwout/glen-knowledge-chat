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


# ── GrooveKart device import (Glen's shipping answers, 2026-07-20) ───────────

IMPORT_TAG = "gk-device-import-2026-07-20"


def _imported():
    return {s: p for s, p in app._PRODUCTS["products"].items()
            if p.get("source") == IMPORT_TAG}


def test_every_imported_device_charges_real_shipping():
    """The money guard. An own-box device with no flat rate cannot be packed by
    quote(), so the cart drops to the coarse qty rule and ships a heavy device
    at a small-box rate — the bug fixed in #1050/#1053. Every imported device
    must resolve to a NON-ZERO shipping charge through the real pricing path.
    """
    ship = {"state": "CA", "country": "US"}
    assert len(_imported()) >= 20
    for slug in _imported():
        out = app._price_cart([{"slug": slug, "qty": 1}], ship=ship)
        assert out.get("shipping_cents", 0) > 0, f"{slug} ships for free"


def test_imported_flat_rates_match_glens_answers():
    exp = {
        "nir-brain-frequency-helmet": 3200,      # "Helmets: we ship - $32 (Large) each"
        "hair-growth-helmet": 3200,
        "photobiomodulation-package": 5500,      # "Large + Medium" = 32 + 23
        "miracule-water-system": 10000,          # "Miracule: $100"
        "air-surface-pro-plus": 3200,            # "Air & Surface PRO+ Large"
        "denas-microcurrent-eye-system": 3200,   # "Denas Large (we ship)"
        "tibetan-singing-bowl-172hz": 10000,     # "Tibetan bowl $100"
        "vagus-nerve-stimulation-kit": 1300,     # "Vegus Nerve kit small"
        "neutralizer-3-pack": 1300,              # "Neutralizer small"
    }
    for slug, cents in exp.items():
        p = app._get_product(slug)
        assert p, slug
        assert p["flat_shipping_cents"] == cents, slug
        assert not p.get("vendor_shipped"), f"{slug}: Glen said WE ship it"


def test_packer_devices_use_bottle_types_prod_actually_has():
    """A bottle_type prod does not recognise silently drops the WHOLE cart to
    the coarse qty rule (see shipping.PROD_BOTTLE_NAMES)."""
    for slug, p in _imported().items():
        bt = p.get("bottle_type")
        assert bt, slug
        if bt != "own-box":
            assert bt in shipping.PROD_BOTTLE_NAMES, f"{slug}: {bt!r} missing in prod"


def test_held_devices_were_not_imported():
    """Held pending Glen: unresolved price or shipping. Importing them blind
    would guess a rate, which undercharges every sale."""
    for slug in ("whole-house-neutralizer", "healing-tools-package",
                 "blue-blocking-photochromic-sunglasses",
                 "breath-tuning-fork-1283hz", "living-water-bottle-filter-refill"):
        assert app._get_product(slug) is None, f"{slug} imported before it was resolved"
