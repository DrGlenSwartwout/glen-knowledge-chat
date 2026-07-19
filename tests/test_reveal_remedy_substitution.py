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


# --- Preference substitutions (Glen 2026-07-19): recommend Y rather than X ---

def test_allerfree_swapped_to_immune_modulation():
    row = {"remedies": [{"name": "AllerFree", "slug": "allerfree", "meaning": "old"}]}
    apply_remedy_substitutions(row)
    r = row["remedies"][0]
    assert r["slug"] == "immune-modulation"   # sellable at /begin/buy/immune-modulation ($69.97)
    assert r["name"] == "Immune Modulation"
    assert "Immune Modulation" in r["meaning"]  # meaning swapped, not left describing AllerFree
    assert "allerfree" not in r["meaning"].lower()


def test_allerfree_by_full_drops_name():
    # matched row may carry the fuller FMP name with an empty/different slug
    row = {"remedies": [{"name": "AllerFree HomeoEnergetic Drops", "slug": ""}]}
    apply_remedy_substitutions(row)
    assert row["remedies"][0]["slug"] == "immune-modulation"


def test_bone_builder_swapped_to_neuro_magnesium():
    row = {"remedies": [{"name": "Bone Builder", "slug": "bone-builder", "meaning": "old"}]}
    apply_remedy_substitutions(row)
    r = row["remedies"][0]
    assert r["slug"] == "neuro-magnesium"   # sellable at /begin/buy/neuro-magnesium ($69.97)
    assert r["name"] == "Neuro-Magnesium"
    assert "Neuro-Magnesium" in r["meaning"]
    assert "bone builder" not in r["meaning"].lower()


def test_bone_builder_by_name_when_slug_differs():
    row = {"remedies": [{"name": "Bone Builder", "slug": "13-bone-builder"}]}
    apply_remedy_substitutions(row)
    assert row["remedies"][0]["slug"] == "neuro-magnesium"


def test_preference_swaps_in_layers_and_idempotent():
    row = {"remedies": [], "layers": [
        {"n": 2, "title": "Immune", "remedy": {"name": "AllerFree", "slug": "allerfree"}},
        {"n": 5, "title": "Bone", "remedy": {"name": "Bone Builder", "slug": "bone-builder"}},
    ]}
    apply_remedy_substitutions(row)
    apply_remedy_substitutions(row)  # second pass: swapped-in slugs are not keys
    assert row["layers"][0]["remedy"]["slug"] == "immune-modulation"
    assert row["layers"][1]["remedy"]["slug"] == "neuro-magnesium"
