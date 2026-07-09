"""A retired duplicate must still map to the storefront, via its live twin.

`aces-eyedrops` and `aces-eye-drops` are the same FMP product (fmp_id 158) under two
slugs. Retiring one with `inactive: true` made `_get_product` return None, which killed
its product page, its checkout, and any old link or order-history slug pointing at it —
even though a perfectly good surviving record exists.

`_superseded()` already knows how to follow the pointer, but only remedy-name resolution
used it (`_live_slug`); the storefront never did. Now `_get_product` follows it, so every
consumer redirects transparently. A record that is inactive with NO successor still
resolves to None — retired means retired.
"""
import importlib
import sys
from pathlib import Path

import pytest


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def test_retired_duplicate_redirects_to_its_live_twin():
    appmod = _app()
    p = appmod._get_product("aces-eyedrops")
    assert p is not None, "retired slug must still map to the storefront"
    assert p["slug"] == "aces-eye-drops", "must carry the SURVIVOR's slug forward"
    assert p["name"] == "ACES Eye Drops"


def test_canonical_slug_is_unchanged():
    appmod = _app()
    p = appmod._get_product("aces-eye-drops")
    assert p and p["slug"] == "aces-eye-drops"
    assert p["bottle_type"] == "5ml"


def test_retired_without_successor_still_resolves_to_none():
    """Retired means retired. Only a `superseded_by` pointer earns a redirect."""
    appmod = _app()
    monkey = {"products": {"dead": {"name": "Dead", "inactive": True},
                           "live": {"name": "Live", "price_cents": 100}}}
    old = appmod._PRODUCTS
    try:
        appmod._PRODUCTS = monkey
        assert appmod._get_product("dead") is None
        assert appmod._get_product("live")["slug"] == "live"
    finally:
        appmod._PRODUCTS = old


def test_unknown_slug_is_none():
    appmod = _app()
    assert appmod._get_product("no-such-slug-at-all") is None


def test_existing_deprecated_pairs_also_redirect():
    """The #730/#732 eye-drop retirements gain the same storefront redirect."""
    appmod = _app()
    p = appmod._get_product("neuro-eye-drops")
    assert p and p["slug"] == "neuro-eye-drops-aces-gl-lite-eye-drops"


def test_aces_pair_is_marked_up_in_the_catalog():
    import json
    P = json.load(open("data/products.json"))["products"]
    assert P["aces-eyedrops"]["inactive"] is True
    assert P["aces-eyedrops"]["superseded_by"] == "aces-eye-drops"
    assert not P["aces-eye-drops"].get("inactive")


def test_cart_line_on_the_retired_slug_prices_and_packs_the_survivor():
    """The money path: a stale slug must not silently drop out of the cart."""
    appmod = _app()
    from dashboard import shipping as S
    p = appmod._get_product("aces-eyedrops")
    assert p["price_cents"] == 6997
    assert S.resolve_bottle_type(p["slug"], p) == "5ml"
