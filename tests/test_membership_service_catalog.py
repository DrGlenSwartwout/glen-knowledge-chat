"""Membership services are selectable, correctly priced invoice catalog lines."""
import json
from pathlib import Path

from dashboard import products


REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_products():
    return json.loads((REPO_ROOT / "data" / "products.json").read_text())["products"]


def test_membership_services_have_invoice_catalog_shape():
    catalog = _repo_products()

    assert catalog["membership"] == {
        "name": "Membership",
        "price_cents": 9900,
        "info_only": True,
        "service": True,
        "source": "manual-add-2026-07-26-membership-services",
        "_note": (
            "Invoiceable individual membership service. Kept out of self-checkout "
            "and shipping while remaining selectable in the in-house order builder."
        ),
    }
    assert catalog["family-membership"] == {
        "name": "Family Membership",
        "price_cents": 14900,
        "info_only": True,
        "service": True,
        "source": "manual-add-2026-07-26-membership-services",
        "_note": (
            "Invoiceable family membership service. Kept out of self-checkout and "
            "shipping while remaining selectable in the in-house order builder."
        ),
    }


def test_membership_services_are_selectable_and_unshippable(monkeypatch):
    monkeypatch.setattr(products, "load_products", _repo_products)

    rows = {
        row["slug"]: row
        for row in products.catalog(with_ingredients_only=False)
    }

    assert rows["membership"]["price_cents"] == 9900
    assert rows["membership"]["service"] is True
    assert rows["membership"]["shippable"] is False
    assert rows["family-membership"]["price_cents"] == 14900
    assert rows["family-membership"]["service"] is True
    assert rows["family-membership"]["shippable"] is False
