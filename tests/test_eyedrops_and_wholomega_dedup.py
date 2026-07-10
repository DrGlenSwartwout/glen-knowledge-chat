"""Remaining catalog gaps after #746.

1. Four ACTIVE eye-drop records still had no `bottle_type`. An unmapped product resolves
   to the "default" placeholder, which quote() cannot pack, so the WHOLE cart it rides in
   silently falls back to the coarse qty rule. Glen: map them to 5ml, like every other
   mapped eye drop.

2. `wholomega` and `wholomega-30-gelcaps` are the same product at the same $69.97 under
   two slugs. Glen: `wholomega` is canonical. Retire the twin with `superseded_by` so the
   storefront, order history and FMP-id lookups all redirect to the survivor (#746 made
   `_get_product` follow that pointer).
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

EYE_DROPS = ("neuro-eye-drops-aces-gl-lite-eye-drops",
             "clear-lens-eye-drops-aces-cat-eye-drops",
             "pterygium-eye-drops",
             "myopia-eye-drops")


def _app():
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _catalog():
    return json.load(open("data/products.json"))["products"]


def test_all_active_eye_drops_are_5ml():
    P = _catalog()
    for slug in EYE_DROPS:
        assert P[slug]["bottle_type"] == "Dropper 5 mL", slug


def test_no_active_eye_drop_is_left_unmapped():
    """Guard the whole family, not just the four named: an eye drop with no bottle type
    poisons every cart it rides in."""
    P = _catalog()
    unmapped = [s for s, p in P.items()
                if "eye drop" in (p.get("name") or "").lower().replace("eyedrop", "eye drop")
                and not p.get("inactive")
                and not (p.get("service") or p.get("info_only"))
                and not p.get("bottle_type")]
    assert unmapped == [], f"active eye drops with no bottle_type: {unmapped}"


def test_wholomega_twin_is_retired_and_points_at_the_survivor():
    P = _catalog()
    assert P["wholomega-30-gelcaps"]["inactive"] is True
    assert P["wholomega-30-gelcaps"]["superseded_by"] == "wholomega"
    assert not P["wholomega"].get("inactive")


def test_retired_wholomega_still_maps_to_the_storefront():
    """#746's `_get_product` -> `_superseded` hop must carry the survivor forward."""
    appmod = _app()
    p = appmod._get_product("wholomega-30-gelcaps")
    assert p is not None
    assert p["slug"] == "wholomega"
    assert p["bottle_type"] == "30 Caps"
    assert p["price_cents"] == 6997


def test_retired_twin_keeps_its_fmp_id_so_fmp_lookups_still_resolve():
    """The survivor has no fmp_id; the retired record keeps 1085. Retired records stay in
    the catalog, so an fmp_id lookup still finds it and redirects. Do NOT delete."""
    P = _catalog()
    assert P["wholomega-30-gelcaps"]["fmp_id"] == "1085"


def test_120_variants_share_the_120_cap_bottle():
    """This PR declared the 120 pair duplicate too (see test_wholomega_120_dedup.py);
    only the shared bottle type still binds both records here."""
    P = _catalog()
    for slug in ("wholomega-120-gelcaps", "wholomega-120-capsules"):
        assert P[slug]["bottle_type"] == "120 caps"
