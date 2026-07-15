import sqlite3
from dashboard import life_stress_curation as cur

PRODUCTS = {"products": {
    "holly-flower-essence-in-terrain-restore": {"name": "Holly Flower Essence in Terrain Restore"},
    "willow-flower-essence-in-terrain-restore": {"name": "Willow Flower Essence in Terrain Restore"},
}}
BLOCK = {"label": "Life Stress", "patterns": [{"emotion": "Fear", "score": 1.0}],
         "items": [{"slug": "mimulus-...", "name": "Mimulus Flower Essence",
                    "url": "/begin/product/mimulus-...", "note": "for the fear pattern in your scan"}]}


def _cx():
    cx = sqlite3.connect(":memory:")
    cur.init_table(cx)
    return cx


def test_no_curation_unchanged():
    out = cur.apply(_cx(), "a@b.com", BLOCK, products=PRODUCTS)
    assert out == BLOCK and "curated" not in out


def test_curation_replaces_items_with_slugs():
    # what the practitioner UI (Task 7) actually saves: resolved slugs
    cx = _cx()
    cur.set(cx, "a@b.com", "42",
            ["holly-flower-essence-in-terrain-restore", "willow-flower-essence-in-terrain-restore"],
            "Take these")
    out = cur.apply(cx, "a@b.com", BLOCK, products=PRODUCTS)
    assert out["curated"] is True
    names = [i["name"] for i in out["items"]]
    assert names == ["Holly Flower Essence in Terrain Restore", "Willow Flower Essence in Terrain Restore"]
    assert out["items"][0]["url"] == "/begin/product/holly-flower-essence-in-terrain-restore"
    assert all(i["note"] == "Take these" for i in out["items"])
    assert out["label"] == "Life Stress"


def test_curation_replaces_items_with_names():
    # robustness: also accept essence NAMES (resolved via slug_for_essence)
    cx = _cx()
    cur.set(cx, "a@b.com", "42", ["Holly Flower Essence", "Willow Flower Essence"], "Take these")
    out = cur.apply(cx, "a@b.com", BLOCK, products=PRODUCTS)
    assert out["curated"] is True
    names = [i["name"] for i in out["items"]]
    assert names == ["Holly Flower Essence in Terrain Restore", "Willow Flower Essence in Terrain Restore"]
    assert out["items"][0]["url"].startswith("/begin/product/")


def test_unresolvable_slug_dropped():
    cx = _cx()
    cur.set(cx, "a@b.com", "42", ["Holly Flower Essence", "Nonexistent Xyz"], "")
    out = cur.apply(cx, "a@b.com", BLOCK, products=PRODUCTS)
    assert len(out["items"]) == 1 and "Holly" in out["items"][0]["name"]


def test_all_unresolvable_returns_original_block():
    cx = _cx()
    cur.set(cx, "a@b.com", "42", ["Nonexistent Xyz", "Also Bogus"], "")
    out = cur.apply(cx, "a@b.com", BLOCK, products=PRODUCTS)
    assert out == BLOCK and "curated" not in out


def test_never_raises(monkeypatch):
    cx = _cx()
    cur.set(cx, "a@b.com", "42", ["Holly Flower Essence"], "")
    monkeypatch.setattr(cur, "slug_for_essence", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cur.apply(cx, "a@b.com", BLOCK, products=PRODUCTS) == BLOCK  # falls back to original
