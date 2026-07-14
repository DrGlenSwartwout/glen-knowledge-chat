from dashboard import life_stress

PRODUCTS = {"products": {
    "mimulus-flower-essence-in-terrain-restore": {"name": "Mimulus Flower Essence in Terrain Restore"},
    "trust-flower-essence-in-terrain-restore": {"name": "Trust Flower Essence in Terrain Restore"},
}}

def test_substring_match():
    assert life_stress.slug_for_essence("Mimulus Flower Essence", PRODUCTS) == \
        "mimulus-flower-essence-in-terrain-restore"

def test_unresolved_returns_empty():
    assert life_stress.slug_for_essence("Nonexistent Essence", PRODUCTS) == ""

def test_blank_never_raises():
    assert life_stress.slug_for_essence("", PRODUCTS) == ""
    assert life_stress.slug_for_essence(None, PRODUCTS) == ""
