"""The WholOmega 120 pair, retired per Glen: `wholomega-120-gelcaps` at $190.00 is canonical.

Unlike the two pairs retired in #748, this one is NOT price-equal: the twin lists $189.97
and the survivor $190.00. Glen picked the survivor's price, so there is no price-equality
assert to make here — only that the redirect lands on the survivor's price.

The load-bearing part is the `ingredients` carry-forward. `_product_card()` falls back to
`_generate_card()` — an LLM — for any product with no `ingredients` array, and Pinecone has
ZERO chunks under either WholOmega 120 title, so nothing grounds that generation. Retiring
the record that holds the real ingredient list, while keeping one that holds none, would
put an INVENTED ingredient list on a supplement page. The array moves to the survivor.

#748 already did exactly that to `aces-eye-drops` (the retired `aces-eyedrops` kept the
ingredients). Same fix, same reason.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

TWIN = "wholomega-120-capsules"
SURVIVOR = "wholomega-120-gelcaps"


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


def test_twin_is_retired_and_points_at_the_survivor():
    P = _catalog()
    assert P[TWIN]["inactive"] is True
    assert P[TWIN]["superseded_by"] == SURVIVOR
    assert not P[SURVIVOR].get("inactive")


def test_both_records_agree_on_bottle_type():
    """A redirect that changes the box is a shipping bug, not a rename."""
    P = _catalog()
    assert P[TWIN]["bottle_type"] == P[SURVIVOR]["bottle_type"] == "120 caps"


def test_retired_slug_resolves_to_the_survivor_at_the_survivors_price():
    """Glen chose $190.00. The twin's $189.97 is superseded, not preserved."""
    appmod = _app()
    p = appmod._get_product(TWIN)
    assert p is not None
    assert p["slug"] == SURVIVOR
    assert p["price_cents"] == 19000


def test_survivor_carries_a_real_ingredient_list_not_an_llm_generated_one():
    """The whole reason the twin's `ingredients` array moves across. Without it,
    `_product_card` -> `_generate_card` invents the ingredients of a supplement."""
    P = _catalog()
    ing = P[SURVIVOR].get("ingredients")
    assert ing, f"{SURVIVOR} has no ingredients array -> its card would be generated"
    names = " ".join((i.get("name") or "") for i in ing).lower()
    assert "docosahexaenoic" in names, names[:120]


def test_aces_survivor_also_carries_its_ingredients():
    """#748 retired `aces-eyedrops`, which held the ingredient list, and kept
    `aces-eye-drops`, which held none. Closing that same gap."""
    P = _catalog()
    assert P["aces-eye-drops"].get("ingredients"), \
        "aces-eye-drops has no ingredients -> LLM-generated list on a supplement page"


def test_fmp_448_maps_to_the_live_slug_not_the_retired_one():
    """`fmp_history` keys purchase_history rows straight off this map — it never calls
    `_get_product`, so the `superseded_by` hop cannot save it. A member's repertoire would
    be seeded with a dead slug and the reorder discount (`slug in repertoire_slugs`, tested
    against the RESOLVED slug) would silently never match."""
    resolved = json.load(open("data/fmp_slug_map.json"))["resolved"]
    assert resolved["448"] == SURVIVOR


def test_retired_twin_stays_in_the_catalog():
    """Retiring never deletes: an fmp_id or old-link lookup must still find the record
    and redirect. Deleting is what breaks FMP linkage."""
    assert TWIN in _catalog()
