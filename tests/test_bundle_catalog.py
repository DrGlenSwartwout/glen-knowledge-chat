import json, os
from dashboard.bundle_pricing import resolve_component

def _catalog():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "products.json")
    with open(path) as f:
        return json.load(f)["products"]

PORTED = [
    "crystalline-lens-program", "gut-terrain-program", "dry-eye-relief-program",
    "macular-wellness-program", "glucose-tolerance-program", "brain-program",
    "scar-reduction-program", "iop-program", "dental-bundle", "sleep-bundle",
]
DEVICE_BUNDLES = {"dental-bundle", "sleep-bundle"}

def test_all_ten_present_and_flagged():
    c = _catalog()
    for slug in PORTED:
        assert slug in c, f"missing {slug}"
        p = c[slug]
        assert p.get("bundle") is True, f"{slug} not bundle:true"
        assert p.get("price_rule") == "components_less_10pct", f"{slug} missing price_rule"
        assert isinstance(p.get("bundle_component_slugs"), list) and p["bundle_component_slugs"], slug

def test_autoship_eligibility():
    c = _catalog()
    for slug in PORTED:
        p = c[slug]
        expected = slug not in DEVICE_BUNDLES
        assert p.get("autoship_eligible") is expected, f"{slug} autoship_eligible wrong"

def test_every_component_slug_resolves():
    c = _catalog()
    for slug in PORTED:
        for comp in c[slug]["bundle_component_slugs"]:
            assert resolve_component(comp["slug"], c) is not None, \
                f"{slug} component {comp['slug']} does not resolve"
