import json, os
from dashboard import life_stress

def test_every_essence_resolves_to_a_slug():
    with open(life_stress._EMOTION_MAP_PATH, encoding="utf-8") as fh:
        m = json.load(fh)
    products = life_stress._load_json(life_stress._PRODUCTS_PATH)
    unresolved = []
    for emotion, essences in m.items():
        for name in essences:
            if not life_stress.slug_for_essence(name, products):
                unresolved.append((emotion, name))
    assert not unresolved, f"dead essence names (no products.json slug): {unresolved}"

def test_map_has_the_core_emotions():
    with open(life_stress._EMOTION_MAP_PATH, encoding="utf-8") as fh:
        m = json.load(fh)
    for e in ["Fear", "Anger", "Grief", "Worry", "Love"]:
        assert e in m and m[e], f"missing/empty emotion: {e}"
