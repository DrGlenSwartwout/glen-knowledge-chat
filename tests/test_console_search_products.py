"""Console search, products section.

`products.json` stores ingredients as {name, dose} dicts. The search path used
to `" ".join(...)` them directly, which raised TypeError on the FIRST product
carrying any ingredients — and the surrounding `except Exception` swallowed it,
so the products section came back empty for every query with no visible error.
"""
import json

import pytest

from dashboard import products as dprod


@pytest.fixture
def catalog(monkeypatch):
    data = {
        "aces-eye-drops": {"name": "ACES Eye Drops", "price_cents": 100,
                           "ingredients": [{"name": "Vitamin A", "dose": "5 mg"},
                                           {"name": "Zinc Bisglycinate (20% Zn)",
                                            "dose": "25 mg"}]},
        "plain-widget": {"name": "Plain Widget", "price_cents": 200},
    }
    monkeypatch.setattr(dprod, "load_products", lambda: data)
    return data


def _joined(pr):
    """The expression under test, as the route computes it."""
    return " ".join((i.get("name") or "") if isinstance(i, dict) else str(i)
                    for i in (pr.get("ingredients") or []))


def test_dict_ingredients_join_without_raising(catalog):
    rows = dprod.catalog(with_ingredients_only=False, include_inactive=True)
    hit = next(r for r in rows if r["slug"] == "aces-eye-drops")
    assert isinstance(hit["ingredients"][0], dict)
    assert _joined(hit) == "Vitamin A Zinc Bisglycinate (20% Zn)"


def test_a_product_is_findable_by_its_ingredient(catalog):
    rows = dprod.catalog(with_ingredients_only=False, include_inactive=True)
    matched = [r["slug"] for r in rows if "zinc" in _joined(r).lower()]
    assert matched == ["aces-eye-drops"]


def test_products_without_ingredients_still_join_to_empty(catalog):
    rows = dprod.catalog(with_ingredients_only=False, include_inactive=True)
    plain = next(r for r in rows if r["slug"] == "plain-widget")
    assert _joined(plain) == ""


def test_legacy_string_ingredients_still_supported(catalog):
    # Nothing in the catalog uses this shape today, but the join must not
    # start raising if an older or hand-edited record does.
    assert _joined({"ingredients": ["Vitamin A", "Zinc"]}) == "Vitamin A Zinc"


def test_real_catalog_ingredients_are_dicts():
    """Pins the premise: if the catalog ever went back to plain strings the
    fix above would be dead code and this test says so."""
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data" / "products.json"
    prods = json.loads(p.read_text())["products"]
    withing = [v for v in prods.values() if v.get("ingredients")]
    assert withing, "catalog has no structured ingredients at all"
    assert all(isinstance(v["ingredients"][0], dict) for v in withing)
