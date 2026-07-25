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


def test_every_product_best_ff_is_null():
    seed = _load()
    for p in seed["products"]:
        assert p.get("best_ff") is None, (
            f"product {p.get('name')!r} has a non-null best_ff ({p.get('best_ff')!r}) "
            "in the committed seed -- best_ff mappings are guesses the generator "
            "must never invent; only Glen supplies them by hand after review")


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
