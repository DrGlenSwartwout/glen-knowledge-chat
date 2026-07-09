"""catalog() exposes the same shippability answer the pricer uses."""
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# load_products() reads a single module-level product catalog that is global,
# mutable state: other tests in the suite monkeypatch or empty it out for their
# own purposes, so by the time these tests run in the full suite the real
# catalog may no longer contain biofield-analysis/neuro-magnesium at all. These
# tests only care about catalog()'s own filtering/shaping logic, so they own a
# small fixture catalog via monkeypatch instead of depending on load order.
_FIXTURE_PRODUCTS = {
    "biofield-analysis": {
        "service": True,
        "info_only": True,
        "bottle_type": None,
        "name": "Biofield Analysis",
        "price_cents": 30000,
    },
    "evox-session": {
        "service": True,
        "info_only": True,
        "bottle_type": None,
        "name": "EVOX Session",
        "price_cents": 15000,
    },
    # Info-only but not a "service" line (e.g. an affiliate/referral item) —
    # still must come back unshippable since there's no bottle to pack.
    "emf-protection-guide": {
        "service": False,
        "info_only": True,
        "bottle_type": None,
        "name": "EMF Protection Guide",
        "price_cents": 0,
    },
    "neuro-magnesium": {
        "service": False,
        "info_only": False,
        "bottle_type": "small",
        "name": "Neuro Magnesium",
        "price_cents": 6997,
    },
}


@pytest.fixture
def fixture_catalog(monkeypatch):
    monkeypatch.setattr("dashboard.products.load_products", lambda: _FIXTURE_PRODUCTS)


def test_catalog_marks_services_unshippable(fixture_catalog):
    from dashboard.products import catalog
    rows = {r["slug"]: r for r in catalog(with_ingredients_only=False, include_inactive=True)}
    assert rows["biofield-analysis"]["shippable"] is False
    assert rows["evox-session"]["shippable"] is False


def test_catalog_marks_a_real_product_shippable(fixture_catalog):
    from dashboard.products import catalog
    rows = {r["slug"]: r for r in catalog(with_ingredients_only=False, include_inactive=True)}
    assert rows["neuro-magnesium"]["shippable"] is True


def test_every_catalog_row_has_shippable(fixture_catalog):
    from dashboard.products import catalog
    rows = catalog(with_ingredients_only=False, include_inactive=True)
    assert rows, "catalog must not be empty"
    assert all(isinstance(r.get("shippable"), bool) for r in rows)
