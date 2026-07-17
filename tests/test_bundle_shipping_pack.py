from dashboard.shipping import bundle_component_products, UnknownBundleComponent

CATALOG = [
    {"slug": "iop-syntropy", "name": "IOP Syntropy", "bottle_type": "30ml"},
    {"slug": "ocuflow-daytime", "name": "OcuFlow Daytime", "bottle_type": "30ml"},
    {"slug": "terrain-restore", "name": "Terrain Restore", "bottle_type": "60ct"},
    {"slug": "microbiome", "name": "Microbiome", "bottle_type": "60ct"},
    {"slug": "fiber-cleanse", "name": "Fiber Cleanse", "bottle_type": "60ct"},
    {"slug": "aces-eye-drops", "name": "ACES Eye Drops", "bottle_type": "15ml"},
]

def test_slugs_preferred_and_qty_expanded():
    # IOP = 3x IOP Syntropy + 2x OcuFlow Daytime -> 5 bottles
    bundle = {"bundle": True, "slug": "iop-program",
              "bundle_component_slugs": [
                  {"slug": "iop-syntropy", "qty": 3},
                  {"slug": "ocuflow-daytime", "qty": 2}]}
    out = bundle_component_products(bundle, CATALOG)
    slugs = [p["slug"] for p in out]
    assert slugs == ["iop-syntropy", "iop-syntropy", "iop-syntropy",
                     "ocuflow-daytime", "ocuflow-daytime"]

def test_slugs_without_bundle_components_do_not_raise():
    # gut-terrain has slugs but NO bundle_components (the Task 2 bug)
    bundle = {"bundle": True, "slug": "gut-terrain-program",
              "bundle_component_slugs": [
                  {"slug": "terrain-restore", "qty": 1},
                  {"slug": "microbiome", "qty": 1},
                  {"slug": "fiber-cleanse", "qty": 1}]}
    out = bundle_component_products(bundle, CATALOG)
    assert [p["slug"] for p in out] == ["terrain-restore", "microbiome", "fiber-cleanse"]

def test_qty_defaults_to_one():
    bundle = {"bundle": True, "slug": "x",
              "bundle_component_slugs": [{"slug": "microbiome"}]}
    out = bundle_component_products(bundle, CATALOG)
    assert [p["slug"] for p in out] == ["microbiome"]

def test_unknown_slug_raises():
    bundle = {"bundle": True, "slug": "x",
              "bundle_component_slugs": [{"slug": "does-not-exist", "qty": 1}]}
    try:
        bundle_component_products(bundle, CATALOG)
        assert False, "expected UnknownBundleComponent"
    except UnknownBundleComponent:
        pass

def test_name_fallback_still_works():
    # a legacy bundle with only bundle_components (no slugs)
    bundle = {"bundle": True, "slug": "dry-eye-relief-program",
              "bundle_components": ["ACES Eye Drops", "Microbiome"]}
    out = bundle_component_products(bundle, CATALOG)
    assert [p["slug"] for p in out] == ["aces-eye-drops", "microbiome"]

def test_non_bundle_returns_empty():
    assert bundle_component_products({"slug": "x"}, CATALOG) == []
