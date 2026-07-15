from dashboard import life_stress_curation as cur

PRODUCTS = {"products": {
    "holly-flower-essence-in-terrain-restore": {"name": "Holly Flower Essence in Terrain Restore"},
}}
BLOCK = {"label":"Life Stress","patterns":[],
         "items":[{"slug":"mimulus-x","name":"Mimulus","url":"/begin/product/mimulus-x","note":"n"}]}

def test_none_curation_unchanged():
    out = cur.apply_data(None, BLOCK, products=PRODUCTS)
    assert out == BLOCK and "curated" not in out

def test_curation_dict_overrides():
    out = cur.apply_data({"slugs":["Holly Flower Essence"],"note":"take it"}, BLOCK, products=PRODUCTS)
    assert out["curated"] is True
    assert "Holly" in out["items"][0]["name"]
    assert out["items"][0]["url"].startswith("/begin/product/")
    assert out["items"][0]["note"] == "take it"

def test_empty_slugs_unchanged():
    assert cur.apply_data({"slugs":[],"note":""}, BLOCK, products=PRODUCTS) == BLOCK

def test_never_raises(monkeypatch):
    monkeypatch.setattr(cur, "slug_for_essence", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cur.apply_data({"slugs":["Holly Flower Essence"]}, BLOCK, products=PRODUCTS) == BLOCK
