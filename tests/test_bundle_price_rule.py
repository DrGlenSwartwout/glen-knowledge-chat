import json, os
from dashboard.bundle_pricing import compute_bundle_price_cents

CATALOG = {
    "aces-eye-drops": {"name": "ACES Eye Drops", "price_cents": 6997},
    "moisturize": {"name": "Moisturize", "price_cents": 6997},
    "wholomega": {"name": "WholOmega", "price_cents": 6997},
    "iop-syntropy": {"name": "IOP Syntropy", "price_cents": 6997},
    "ocuflow-daytime": {"name": "OcuFlow Daytime", "price_cents": 6997},
}

def test_sum_less_10pct_simple():
    bundle = {"bundle": True, "price_rule": "components_less_10pct",
              "bundle_component_slugs": [
                  {"slug": "aces-eye-drops", "qty": 1},
                  {"slug": "moisturize", "qty": 1},
                  {"slug": "wholomega", "qty": 1}]}
    # 3 * 6997 = 20991 ; * 0.9 = 18891.9 -> 18892
    assert compute_bundle_price_cents(bundle, CATALOG) == 18892

def test_sum_less_10pct_with_qty():
    bundle = {"bundle": True, "price_rule": "components_less_10pct",
              "bundle_component_slugs": [
                  {"slug": "iop-syntropy", "qty": 3},
                  {"slug": "ocuflow-daytime", "qty": 2}]}
    # 5 * 6997 = 34985 ; * 0.9 = 31486.5 -> 31486  (banker's rounding of .5 -> even)
    assert compute_bundle_price_cents(bundle, CATALOG) == 31486

def test_unknown_component_raises():
    bundle = {"bundle": True, "price_rule": "components_less_10pct",
              "bundle_component_slugs": [{"slug": "does-not-exist", "qty": 1}]}
    try:
        compute_bundle_price_cents(bundle, CATALOG)
        assert False, "expected KeyError"
    except KeyError:
        pass

def test_live_catalog_prices_match_rule():
    """Drift guard: every components_less_10pct bundle in the real catalog
    stores exactly the rule price."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "products.json")
    with open(path) as f:
        data = json.load(f)
    products = data["products"]
    checked = 0
    for slug, p in products.items():
        if p.get("price_rule") == "components_less_10pct":
            expected = compute_bundle_price_cents(p, products)
            assert p.get("price_cents") == expected, (
                f"{slug}: stored {p.get('price_cents')} != rule {expected}")
            checked += 1
    assert checked >= 10, f"expected >=10 rule-priced bundles, saw {checked}"
