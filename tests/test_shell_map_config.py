import json
from pathlib import Path
import begin_funnel
import shell_nav

CFG = Path(__file__).resolve().parent.parent / "static" / "shell-map.json"


def _land_keys():
    return [s["key"] for s in begin_funnel.JOURNEY_STEPS]


def test_shipped_config_is_valid():
    cfg = json.loads(CFG.read_text())
    assert shell_nav.validate_shell_map(cfg, _land_keys()) == []


def test_validator_flags_unknown_land():
    bad = {"lands": {"scan": {"name": "x", "category": "scan", "intrigue": "y"},
                     "BOGUS": {"name": "z", "category": "scan", "intrigue": "y"}},
           "categories": {"scan": {"icon": "🌀"}}}
    errs = shell_nav.validate_shell_map(bad, ["scan", "find", "heal", "give"])
    assert any("BOGUS" in e for e in errs)


def test_validator_flags_missing_category_style():
    bad = {"lands": {"scan": {"name": "x", "category": "missing", "intrigue": "y"}},
           "categories": {}}
    errs = shell_nav.validate_shell_map(bad, ["scan", "find", "heal", "give"])
    assert any("missing" in e for e in errs)


def test_featured_slugs_exist_in_catalog():
    import json
    from dashboard import products as _products
    cfg = json.loads(CFG.read_text())
    catalog = set(_products.load_products().keys())
    missing = []
    for key, land in cfg["lands"].items():
        f = land.get("featured")
        if f and f["product_slug"] not in catalog:
            missing.append(f["product_slug"])
    assert missing == [], f"featured slugs not in products.json: {missing}"
