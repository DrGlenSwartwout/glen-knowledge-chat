"""Curated remedy substitution on the reveal client read path: an unpurchasable
matched remedy ("Relax", 404 on /begin/buy/relax) is swapped for its sellable
equivalent ("Stress Release", /begin/buy/stress-release resolves) everywhere it
renders with an order button — remedies list AND layer remedies, so display,
entitlement, pricing, and checkout stay in sync.
"""
from dashboard.biofield_reveals import apply_remedy_substitutions


def test_relax_slug_swapped_to_stress_release():
    row = {"remedies": [{"name": "Relax", "slug": "relax", "meaning": "old"}]}
    apply_remedy_substitutions(row)
    r = row["remedies"][0]
    assert r["slug"] == "stress-release"
    assert r["name"] == "Stress Release"
    assert "Stress Release" in r["meaning"]  # meaning swapped, not left describing Relax


def test_relax_by_name_when_slug_differs():
    row = {"remedies": [{"name": "Relax", "slug": "", "meaning": "x"}]}
    apply_remedy_substitutions(row)
    assert row["remedies"][0]["slug"] == "stress-release"


def test_layer_remedy_swapped_too():
    row = {"remedies": [], "layers": [
        {"n": 3, "title": "Calm", "remedy": {"name": "Relax", "slug": "relax"}},
        {"n": 4, "title": "Liver", "remedy": {"name": "Liver Support", "slug": "liver-support"}},
    ]}
    apply_remedy_substitutions(row)
    assert row["layers"][0]["remedy"]["slug"] == "stress-release"
    assert row["layers"][1]["remedy"]["slug"] == "liver-support"  # untouched


def test_non_matching_untouched():
    row = {"remedies": [{"name": "Heart Health", "slug": "heart-health"}]}
    apply_remedy_substitutions(row)
    assert row["remedies"][0]["slug"] == "heart-health"


def test_idempotent():
    row = {"remedies": [{"name": "Relax", "slug": "relax"}]}
    apply_remedy_substitutions(row)
    apply_remedy_substitutions(row)  # second pass: stress-release is not a key
    assert row["remedies"][0]["slug"] == "stress-release"


def test_safe_on_none_and_malformed():
    assert apply_remedy_substitutions(None) is None
    # missing keys / non-dict remedies must not raise
    apply_remedy_substitutions({"remedies": [None, "x", {}], "layers": [None, {"remedy": None}]})
