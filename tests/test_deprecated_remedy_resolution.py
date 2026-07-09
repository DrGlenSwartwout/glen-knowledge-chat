"""A deprecated duplicate must never end up on a causal chain as a dead end.

#730 retired two duplicate eye-drop records (`inactive: true`) and renamed the
survivors. But FMP still marks all four active, and `resolve_remedy_name` fuzzy-matches
what Glen narrates against FMP names -- not catalog names. FMP holds:

    'Neuro+ Eye Drops'                          <- retired (slug neuro-eye-drops)
    'Neuro Eye Drops\\nACES+GL Lite Eye Drops'   <- live    (slug neuro-eye-drops-aces-gl-lite-...)

so saying "neuro eye drops" -- the survivor's own display name -- fuzzy-matched the
SHORT retired name, resolving to an unsellable slug that was carried forward silently
(it is truthy, so it never reached the `dropped` list).

Two guards, tested here:
  1. resolve_remedy_name excludes catalog-`inactive` products from the candidate pool.
  2. _resolve_remedy_slug follows `superseded_by` to the surviving twin.
"""
import json
import sqlite3
from pathlib import Path

import pytest

from dashboard.biofield_authoring import (
    resolve_remedy_name, _deprecated_catalog_names, _sellable_names, _norm_name,
)

ROOT = Path(__file__).resolve().parent.parent

LIVE_NEURO = "neuro-eye-drops-aces-gl-lite-eye-drops"
LIVE_CLEAR = "clear-lens-eye-drops-aces-cat-eye-drops"
DEAD_NEURO = "neuro-eye-drops"
DEAD_CLEAR = "clear-lens-eye-drops-aces-cat-eye-drops-2"

FMP_NEURO_LIVE = "Neuro Eye Drops\nACES+GL Lite Eye Drops"
FMP_NEURO_DEAD = "Neuro+ Eye Drops"


def _products():
    return json.loads((ROOT / "data" / "products.json").read_text())["products"]


def _fmp_db():
    cx = sqlite3.connect(":memory:")
    cx.execute("CREATE TABLE fmp_snap_products (product_name TEXT, active TEXT)")
    cx.executemany(
        "INSERT INTO fmp_snap_products(product_name, active) VALUES (?, 'Yes')",
        [(FMP_NEURO_LIVE,), (FMP_NEURO_DEAD,),
         ("Clear Lens Eye Drops ACES+CAT Eye Drops",),
         ("Clear Lens+ Eye Drops ACES+CAT Eye Drops",)])
    cx.commit()
    return cx


# --- guard 1: deprecated products leave the fuzzy-match pool -------------------

def test_deprecated_names_include_retired_pinecone_titles():
    dep = _deprecated_catalog_names()
    assert _norm_name(FMP_NEURO_DEAD) in dep
    assert _norm_name("Clear Lens+ Eye Drops ACES+CAT Eye Drops") in dep
    # the survivors must NOT be filtered out
    assert _norm_name(FMP_NEURO_LIVE) not in dep
    assert _norm_name("Clear Lens Eye Drops ACES+CAT Eye Drops") not in dep


def test_sellable_names_drops_only_the_retired():
    kept = _sellable_names([FMP_NEURO_LIVE, FMP_NEURO_DEAD])
    assert kept == [FMP_NEURO_LIVE]


def test_saying_the_survivors_name_no_longer_hits_the_retired_twin():
    """THE regression. Before the fix this returned 'Neuro+ Eye Drops'."""
    got = resolve_remedy_name(_fmp_db(), "neuro eye drops")
    assert got != FMP_NEURO_DEAD, "resolved onto the retired duplicate"
    assert got.startswith("Neuro Eye Drops")


def test_norm_name_collapses_newlines_and_discontinue_asterisk():
    assert _norm_name("Neuro Eye Drops\nACES+GL Lite") == "neuro eye drops aces+gl lite"
    assert _norm_name("D-Mannose Syntropy*") == "d-mannose syntropy"


def test_missing_catalog_file_filters_nothing(monkeypatch):
    """A missing/broken products.json must not silently empty the candidate pool."""
    import dashboard.biofield_authoring as ba
    ba._deprecated_catalog_names.cache_clear()
    monkeypatch.setattr(ba, "_PRODUCTS_JSON", "/nonexistent/products.json")
    try:
        assert ba._deprecated_catalog_names() == frozenset()
        assert ba._sellable_names(["Anything"]) == ["Anything"]
    finally:
        ba._deprecated_catalog_names.cache_clear()


# --- guard 2: superseded_by redirects to the surviving twin --------------------

def test_retired_records_point_at_their_live_twin():
    prods = _products()
    assert prods[DEAD_NEURO]["superseded_by"] == LIVE_NEURO
    assert prods[DEAD_CLEAR]["superseded_by"] == LIVE_CLEAR
    for dead in (DEAD_NEURO, DEAD_CLEAR):
        assert prods[dead]["inactive"] is True
    for live in (LIVE_NEURO, LIVE_CLEAR):
        assert not prods[live].get("inactive")
        assert "superseded_by" not in prods[live]


def test_superseded_target_is_always_a_live_product():
    """A pointer into another inactive record would be a dead end one hop later."""
    prods = _products()
    for slug, p in prods.items():
        tgt = p.get("superseded_by")
        if tgt:
            assert tgt in prods, f"{slug} -> unknown slug {tgt}"
            assert not prods[tgt].get("inactive"), f"{slug} -> inactive {tgt}"
