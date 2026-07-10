"""Every order link points at the NEW-STYLE in-funnel product page.

`products.json`'s `url` field is the OLD GrooveKart page and is absent on 669 of 966
sellable products. `/begin/product/<slug>` renders from catalog data for all of them,
keeps the client inside the funnel where the upgrade CTA lives, and honours their
courtesy pricing. So the rule is: never link to remedymatch.com.
"""
import json
from pathlib import Path

from dashboard.order_destination import destination_for


def test_a_slug_becomes_a_new_style_product_page():
    assert destination_for("ed6-heart-driver") == "/begin/product/ed6-heart-driver"


def test_a_blank_slug_yields_no_link():
    assert destination_for("") == ""
    assert destination_for(None) == ""


def test_the_destination_is_never_the_old_storefront():
    for slug in ("es1-lymph", "bfa-big-field-aligner-infoceutical", "mb1-brain-stem-hologram"):
        assert "remedymatch.com" not in destination_for(slug)


def test_it_works_for_a_product_that_has_no_storefront_url():
    """bfa-big-field-aligner-infoceutical is `no_groovekart` with no `url` — and still
    has a new-style page. This is exactly why we do not read the `url` field."""
    p = json.loads((Path(__file__).resolve().parent.parent / "data" / "products.json").read_text())["products"]
    rec = p["bfa-big-field-aligner-infoceutical"]
    assert not rec.get("url")
    assert destination_for("bfa-big-field-aligner-infoceutical") == "/begin/product/bfa-big-field-aligner-infoceutical"
