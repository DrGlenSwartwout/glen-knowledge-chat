"""One bottle vocabulary, and it is prod's.

Prod's `bottle_types` library was hand-built through /admin/shipping with names like
'30 Caps' and 'Dropper 5 mL'. The repo used its own private names ('30cap', '5ml') and
`_STANDARD_BOTTLES` only seeds a FRESH catalog, so prod never received them. Result: a
`bottle_type` written into products.json was INERT in prod — `pick_boxes` raised
UnknownBottleType on the unknown name, `quote()` returned no rate, and the whole cart
silently fell back to the coarse qty rule. Green tests, zero production effect, because
the test fixture seeded the repo's own vocabulary.

So: the repo now speaks prod's names. Verified against the live library 2026-07-09.
"""
import json

from dashboard.shipping import (PENDING_BOTTLE_NAMES, PROD_BOTTLE_NAMES,
                                _STANDARD_BOTTLES)


def _baselines():
    P = json.load(open("data/products.json"))["products"]
    return {s: p["bottle_type"] for s, p in P.items() if p.get("bottle_type")}


def test_prod_library_is_recorded_verbatim():
    """The twenty-two names live in prod's bottle_types table (GET /api/shipping/bottles).
    '100ml','15ml','30roll','handcradle' were created 2026-07-09; 'toothbrush' (id 17)
    and the device/accessory types 'harmony-laser','dowsing-rods','own-box' (ids 18-20)
    were created 2026-07-10, as were 'book','nasal-clip','denas' (ids 21-23) for the
    book/device batch add; 'nightlight' (id 24) was created 2026-07-11 for the two
    nightlight SKUs. Before each was created the catalog referenced bottles prod
    did not have,
    and those products silently fell back to the qty rule."""
    assert PROD_BOTTLE_NAMES == frozenset({
        "30 Caps", "120 caps", "180 caps", "360 caps", "30 g", "120 g",
        "30ml", "Dropper 5 mL", "Dropper 30 mL", "Dropper 50 mL",
        "100ml", "15ml", "30roll", "handcradle", "toothbrush",
        "harmony-laser", "dowsing-rods", "own-box",
        "book", "nasal-clip", "denas", "nightlight"})


def test_every_catalog_baseline_speaks_a_name_prod_knows():
    """A baseline prod cannot resolve is exactly as broken as no baseline at all."""
    unknown = {s: bt for s, bt in _baselines().items()
               if bt not in PROD_BOTTLE_NAMES and bt not in PENDING_BOTTLE_NAMES}
    assert unknown == {}, f"bottle types prod has never heard of: {unknown}"


def test_nothing_is_pending_anymore():
    """The four missing types were created in prod on 2026-07-09, so the catalog and the
    live library now speak the same fourteen names. A non-empty set here means someone
    introduced a bottle the packer cannot resolve."""
    assert PENDING_BOTTLE_NAMES == frozenset()


def test_the_two_former_orphans_now_resolve_against_prod():
    b = _baselines()
    assert b["neem-oil-rollon"] == "30roll"
    assert b["hand-cradle"] == "handcradle"
    assert {"30roll", "handcradle"} <= PROD_BOTTLE_NAMES


def test_standard_bottles_seed_prods_vocabulary():
    """The fixture must seed the names PROD uses, or a green test proves nothing."""
    seeded = {n for n, _desc, _d, _h in _STANDARD_BOTTLES}
    assert PROD_BOTTLE_NAMES <= seeded, f"missing from seed: {PROD_BOTTLE_NAMES - seeded}"


def test_no_legacy_repo_names_remain_in_the_catalog():
    legacy = {"30cap", "120cap", "5ml", "50ml", "30g"}
    used = set(_baselines().values())
    assert not (used & legacy), f"legacy repo names still in products.json: {used & legacy}"


def test_the_five_renamed_families_landed_on_prod_names():
    b = _baselines()
    assert b["macular-wellness-lycopene"] == "30 Caps"
    assert b["aces-eye-drops"] == "Dropper 5 mL"
    assert b["wholomega"] == "30 Caps"
    assert b["wholomega-120-gelcaps"] == "120 caps"
    assert b["dental-powder"] == "30 g"
