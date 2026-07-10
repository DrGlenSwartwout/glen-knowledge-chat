# tests/test_scan_recommendations_read.py  (part 1 of 2 — read helpers land in Task 3)
"""BFA is rank 1 on 161 scans and resolves to nothing.

69 of 70 infoceutical codes resolve because the catalog's storefront twin carries the
bare code as its `pinecone_title` (es1-lymph -> "ES1"). Both BFA records carry long
titles, so the bare code matches neither. A new `aliases` list fixes it without
touching `pinecone_title`, which would orphan the product's Pinecone vector.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BFA_SLUG = "bfa-big-field-aligner-infoceutical"


def _app():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


def _products():
    return json.loads((ROOT / "data" / "products.json").read_text())["products"]


def test_the_bfa_record_carries_the_bare_code_as_an_alias():
    assert _products()[BFA_SLUG]["aliases"] == ["BFA"]


def test_the_aliased_record_is_the_one_with_a_bottle_type():
    """The other BFA twin has no bottle_type; ordering it resolves the packer's
    'default' bottle, which poisons the shipping quote."""
    rec = _products()[BFA_SLUG]
    assert rec["bottle_type"] == "30ml"
    assert rec.get("description")


def test_pinecone_title_is_untouched():
    assert _products()[BFA_SLUG]["pinecone_title"] == "BFA Big Field Aligner Infoceutical"


def test_the_bare_code_bfa_now_resolves_to_a_live_product():
    app = _app()
    slug = app._resolve_remedy_slug({"name": "BFA"})
    assert slug == BFA_SLUG
    assert app._get_product(slug)


def test_the_other_infoceutical_codes_still_resolve():
    app = _app()
    for code, expected in (("ED6", "ed6-heart-driver"), ("ES7", "es7-muscle"),
                           ("ES1", "es1-lymph"), ("MB1", "mb1-brain-stem-hologram")):
        assert app._resolve_remedy_slug({"name": code}) == expected


def test_mihealth_codes_still_resolve_to_nothing():
    """ER/MR are device cycles, not products. Resolving them would be the bug."""
    app = _app()
    for code in ("ER2", "ER18", "MR4", "MR6"):
        assert not app._resolve_remedy_slug({"name": code})


def test_the_two_duplicate_title_keys_still_resolve_as_before():
    """`_TITLE_TO_SLUG` is a dict comprehension: on a duplicate key the LAST product wins.
    Two duplicate title keys exist. Rebuilding that dict with setdefault would flip them."""
    app = _app()
    assert app._resolve_remedy_slug({"name": "Brain Boost Nootropic"})
    assert app._resolve_remedy_slug({"name": "Forgiveness Flower Essence in Terrain Restore"})


def test_an_alias_never_shadows_a_real_product_name():
    """A collision would silently hand one product's code to another."""
    p = _products()
    names = {(r.get("pinecone_title") or r.get("name") or "").strip().lower() for r in p.values()}
    for slug, rec in p.items():
        for a in rec.get("aliases") or []:
            assert a.strip().lower() not in names, f"{slug} alias {a!r} collides with a product title"
