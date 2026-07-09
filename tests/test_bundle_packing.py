"""A bundle ships as its CONTENTS, not as one mystery item.

`Macular Wellness Program` is one catalog line holding five physical bottles. It has
no `bottle_type` (it isn't a bottle), so `resolve_bottle_type` returned the "default"
placeholder, `quote()` gave up, and `_shipping_for_cart` fell back to the coarse
qty rule counting the bundle as ONE bottle -> Small box, $13.00. The same five bottles
bought individually count as five -> Medium, $23.00. So the bundle undercharged, and
worse, told Rae to pack five bottles into a Small box that cannot hold them.

The catalog already records the contents in `bundle_components` — a field written by
scripts/apply_worklist.py and read by NOTHING. This makes it load-bearing.

Component names are display names, not slugs, so they must be resolved. Resolution is
EXACT and case-insensitive, never fuzzy: on a money path a near-name could pack and
bill the wrong SKU. An unresolvable component fails LOUDLY — a bundle whose contents
cannot be identified cannot be packed correctly by a human either.
"""
import pytest

from dashboard.shipping import UnknownBundleComponent, bundle_component_products

A = {"slug": "comp-a", "name": "Comp A", "bottle_type": "30ml"}
B = {"slug": "comp-b", "name": "Comp B", "bottle_type": "30cap"}
CATALOG = [A, B, {"slug": "other", "name": "Other"}]

BUNDLE = {"slug": "bund", "name": "Bundle", "bundle": True,
          "bundle_components": ["Comp A", "Comp B"]}


def test_resolves_components_in_order():
    assert bundle_component_products(BUNDLE, CATALOG) == [A, B]


def test_match_is_case_and_whitespace_insensitive():
    b = dict(BUNDLE, bundle_components=["  comp a ", "COMP B"])
    assert bundle_component_products(b, CATALOG) == [A, B]


def test_non_bundle_has_no_components():
    assert bundle_component_products({"slug": "x", "name": "X"}, CATALOG) == []
    assert bundle_component_products(None, CATALOG) == []


def test_unresolvable_component_fails_loudly():
    """Silently falling back would undercharge and misprint the box — exactly the bug."""
    b = dict(BUNDLE, bundle_components=["Comp A", "Ghost Product"])
    with pytest.raises(UnknownBundleComponent) as e:
        bundle_component_products(b, CATALOG)
    assert "Ghost Product" in str(e.value)


def test_no_fuzzy_matching():
    """A near-name must NOT resolve: 'Comp' is not 'Comp A'. On a money path a fuzzy
    hit could pack and bill a different SKU (ES1 vs ES13)."""
    b = dict(BUNDLE, bundle_components=["Comp"])
    with pytest.raises(UnknownBundleComponent):
        bundle_component_products(b, CATALOG)


def test_bundle_flag_without_components_fails_loudly():
    """A bundle we cannot see inside must never quietly ship as one bottle."""
    b = {"slug": "bund", "name": "Bundle", "bundle": True}
    with pytest.raises(UnknownBundleComponent):
        bundle_component_products(b, CATALOG)


def test_real_catalog_bundles_all_resolve():
    """Guard the live data: every shipped bundle's components must resolve today, or
    the loud failure would block real orders."""
    import json
    P = json.load(open("data/products.json"))["products"]
    catalog = [dict(p, slug=s) for s, p in P.items()]
    bundles = [dict(p, slug=s) for s, p in P.items() if p.get("bundle")]
    assert bundles, "expected bundle products in the catalog"
    for b in bundles:
        comps = bundle_component_products(b, catalog)
        assert len(comps) == len(b["bundle_components"]), b["slug"]
