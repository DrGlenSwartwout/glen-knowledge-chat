"""Asserts invariants of the COMMITTED data/fullscript_seed.json itself.

All other fullscript tests (test_fullscript_module.py, test_fullscript_resolver.py)
use inline fixtures and never touch the real seed file -- so a future bad
regeneration of data/fullscript_seed.json would not be caught by pytest at
all. This file closes that gap. No network access; it only reads the file
already committed to the repo.
"""
import json
import os
import sqlite3

from dashboard import fullscript as fs

SEED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "data", "fullscript_seed.json")


def _load():
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_top_level_keys_exist():
    seed = _load()
    assert "products" in seed
    assert "focus_area_products" in seed
    assert "focus_area_items" in seed


def test_every_product_has_required_non_empty_fields():
    seed = _load()
    assert seed["products"], "no products in the committed seed"
    for p in seed["products"]:
        assert p.get("name"), f"product missing non-empty name: {p!r}"
        assert p.get("external_id"), (
            f"product {p.get('name')!r} has an empty external_id -- later "
            "phases build product deep links from external_id, so this is a "
            "real defect, not a cosmetic gap")
        assert p.get("product_slug"), (
            f"product {p.get('name')!r} has an empty product_slug")


def test_best_ff_is_null_or_nonempty_string():
    """The generator must never invent best_ff mappings -- but Glen may supply
    one deliberately by hand after review (see
    test_magnesium_taurate_maps_to_neuro_magnesium). This guard still catches
    junk: an empty string or whitespace-only best_ff, or a non-string value,
    is never valid -- only None or a real non-empty string."""
    seed = _load()
    for p in seed["products"]:
        best_ff = p.get("best_ff")
        if best_ff is None:
            continue
        assert isinstance(best_ff, str) and best_ff.strip(), (
            f"product {p.get('name')!r} has a junk best_ff ({best_ff!r}) in the "
            "committed seed -- must be None or a non-empty string")


def test_magnesium_taurate_maps_to_neuro_magnesium():
    """Pins Glen's deliberate mapping: Jarrow Magnesium Taurate is his own
    documented stearate-free counterpart to Neuro Magnesium. Guards against a
    silent regression wiping out the one hand-supplied best_ff in the seed."""
    seed = _load()
    matches = [p for p in seed["products"] if p.get("name") == "Magnesium Taurate"]
    assert matches, "Magnesium Taurate product missing from the committed seed"
    p = matches[0]
    assert p.get("best_ff") == "Neuro Magnesium"
    assert p.get("relation") == "substitute"


def test_l_theanine_maps_to_neuro_magnesium():
    """Pins Glen's deliberate L-Theanine mapping (Neuro Magnesium carries the
    same active per his knowledge base; he chose it as the first pick). Guards
    the second hand-supplied best_ff against a silent regression."""
    seed = _load()
    matches = [p for p in seed["products"] if p.get("name") == "L-Theanine"]
    assert matches, "L-Theanine product missing from the committed seed"
    p = matches[0]
    assert p.get("best_ff") == "Neuro Magnesium"
    assert p.get("relation") == "substitute"


def test_focus_area_items_non_empty():
    seed = _load()
    assert seed["focus_area_items"], (
        "focus_area_items is empty -- this makes the E4L scan matcher "
        "structurally dead (no scan code maps to any focus area). This exact "
        "bug already shipped once.")


def test_focus_area_products_reference_real_products():
    seed = _load()
    product_names = {p["name"] for p in seed["products"]}
    for fp in seed["focus_area_products"]:
        assert fp["fs_product_name"] in product_names, (
            f"focus_area_products references {fp['fs_product_name']!r}, "
            "which does not exist in products")


def test_sync_from_seed_counts_match_file_contents():
    seed = _load()
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    fs.init_tables(cx)
    counts = fs.sync_from_seed(cx, seed)

    assert counts["products"] == len(seed["products"])
    assert counts["focus_area_products"] == len(seed["focus_area_products"])
    assert counts["focus_area_items"] == len(seed["focus_area_items"])
    assert counts["condition_products"] == len(seed.get("condition_products", []))

    assert (cx.execute("SELECT COUNT(*) FROM fullscript_products").fetchone()[0]
            == len(seed["products"]))
    assert (cx.execute("SELECT COUNT(*) FROM fullscript_focus_area_products")
            .fetchone()[0] == len(seed["focus_area_products"]))
    assert (cx.execute("SELECT COUNT(*) FROM fullscript_focus_area_items")
            .fetchone()[0] == len(seed["focus_area_items"]))


def test_every_best_ff_resolves_to_a_real_slug():
    """A best_ff that doesn't resolve renders a chip with no link -- a dead chip
    (the NeurOmega mistake: a PRL-crosswalk label that isn't a real product name).
    Every non-null best_ff in the committed seed must resolve to a real slug."""
    import os
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    os.environ.setdefault("PINECONE_API_KEY", "dummy")
    import app
    for p in _load()["products"]:
        ff = p.get("best_ff")
        if ff:
            v = app._fullscript_ff_view(ff, p.get("relation"))
            assert v and v.get("slug"), \
                f"{p['name']}: best_ff {ff!r} does not resolve to a slug (dead chip)"
