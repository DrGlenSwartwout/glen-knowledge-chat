"""TDD for dashboard.orders.physical_units — total physical shipping units
(bottles) in an order's line items. Bundles expand to their component bottles;
services / info-only lines count 0. Pure, no db/app import (bare pytest)."""
from dashboard.orders import physical_units

CATALOG = {
    "bottle-a": {"slug": "bottle-a", "name": "Bottle A"},
    "bottle-b": {"slug": "bottle-b", "name": "Bottle B"},
    "svc-consult": {"slug": "svc-consult", "name": "Consult", "service": True},
    "pdf-guide": {"slug": "pdf-guide", "name": "PDF Guide", "info_only": True},
    "bundle-x": {
        "slug": "bundle-x", "name": "Bundle X", "bundle": True,
        "bundle_component_slugs": [
            {"slug": "bottle-a", "qty": 2},
            {"slug": "bottle-b", "qty": 1},
        ],
    },
    "bundle-broken": {
        "slug": "bundle-broken", "name": "Bundle Broken", "bundle": True,
        "bundle_component_slugs": [
            {"slug": "does-not-exist", "qty": 2},
            {"slug": "bottle-b", "qty": 1},
        ],
    },
}


def test_normal_product_line_counts_qty():
    items = [{"slug": "bottle-a", "qty": 3, "name": "Bottle A"}]
    assert physical_units(items, CATALOG) == 3


def test_bundle_expands_to_components_times_qty():
    # bundle-x = 2x bottle-a + 1x bottle-b = 3 bottles per bundle unit; qty 2 bundles -> 6
    items = [{"slug": "bundle-x", "qty": 2, "name": "Bundle X"}]
    assert physical_units(items, CATALOG) == 6


def test_service_line_counts_zero():
    items = [{"slug": "svc-consult", "qty": 1, "name": "Consult"}]
    assert physical_units(items, CATALOG) == 0


def test_info_only_line_counts_zero():
    items = [{"slug": "pdf-guide", "qty": 1, "name": "PDF Guide"}]
    assert physical_units(items, CATALOG) == 0


def test_unknown_slug_counts_qty_since_is_shippable_empty_is_true():
    items = [{"slug": "ghost-slug", "qty": 4, "name": "Ghost"}]
    assert physical_units(items, CATALOG) == 4


def test_missing_or_zero_qty_is_skipped():
    items = [
        {"slug": "bottle-a", "qty": 0, "name": "Bottle A"},
        {"slug": "bottle-a", "name": "Bottle A (no qty key)"},
        {"slug": "bottle-a", "qty": None, "name": "Bottle A (none qty)"},
    ]
    assert physical_units(items, CATALOG) == 0


def test_mixed_order_sums_correctly():
    items = [
        {"slug": "bottle-a", "qty": 2, "name": "Bottle A"},
        {"slug": "svc-consult", "qty": 1, "name": "Consult"},
        {"slug": "bundle-x", "qty": 1, "name": "Bundle X"},
        {"slug": "pdf-guide", "qty": 5, "name": "PDF Guide"},
    ]
    # 2 (bottle-a) + 0 (service) + 3 (bundle-x = 2+1) + 0 (info_only) = 5
    assert physical_units(items, CATALOG) == 5


def test_bundle_with_unresolvable_component_falls_back_to_declared_qtys():
    # bundle-broken has an unknown component slug; must not raise, falls back
    # to the sum of declared component qtys (2 + 1 = 3) per bundle unit.
    items = [{"slug": "bundle-broken", "qty": 1, "name": "Bundle Broken"}]
    assert physical_units(items, CATALOG) == 3


def test_empty_items_and_none_catalog():
    assert physical_units([], CATALOG) == 0
    assert physical_units(None, CATALOG) == 0
    assert physical_units([{"slug": "bottle-a", "qty": 2}], None) == 2
