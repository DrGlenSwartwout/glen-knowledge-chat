from scripts.populate_bottle_types import classify_from_fmp, family_rule, build_assignments

def test_classify_liquid_and_caps_and_powder():
    assert classify_from_fmp({"zc_sold_display": "50ml", "sold_measurement": "ml", "type": ""}) == "Dropper 50 mL"
    assert classify_from_fmp({"zc_sold_display": "30pullulan", "sold_measurement": "pullulan", "type": ""}) == "30 Caps"
    assert classify_from_fmp({"zc_sold_display": "120vegicaps", "sold_measurement": "vegicaps", "type": ""}) == "120 caps"
    assert classify_from_fmp({"zc_sold_display": "30g", "sold_measurement": "g", "type": "Pure Powders"}) == "120 caps"
    assert classify_from_fmp({"zc_sold_display": "30g", "sold_measurement": "g", "type": "Functional Formulation"}) == "30 g"
    assert classify_from_fmp({"zc_sold_display": "1000ml", "sold_measurement": "ml", "type": ""}) is None  # bulk -> review

def test_family_rule_infoceutical_and_eyedrops():
    assert family_rule("ei8-x", {"name": "EI8 Microbes", "source": "infoceutical-catalog"}) == "30ml"
    assert family_rule("mb1-x", {"name": "MB1 Brain Stem Hologram"}) == "30ml"
    assert family_rule("drops", {"name": "ACES Eyedrops"}) == "Dropper 5 mL"
    assert family_rule("z", {"name": "Quercetin"}) is None

def test_build_assignments_priority_and_review():
    products = {
        "ei8": {"name": "EI8 Microbes", "source": "infoceutical-catalog"},
        "cap": {"name": "Brain Boost"},
        "mystery": {"name": "Mystery Tonic"},
        "already": {"name": "Foo", "bottle_type": "15ml"},
    }
    fmp = {"brain boost": {"zc_sold_display": "30pullulan", "sold_measurement": "pullulan", "type": ""}}
    m = build_assignments(products, fmp)
    assert m["assignments"]["ei8"] == "30ml"     # family rule
    assert m["assignments"]["cap"] == "30 Caps"    # fmp join
    assert "already" not in m["assignments"]      # never overwrite
    assert any(r["slug"] == "mystery" for r in m["review"])  # unmatched -> review

# ── New tests for recovery steps ──────────────────────────────────────────────

def test_family_rule_rollon():
    """Roll-on products (all name/description variants) → 30roll."""
    assert family_rule("neem-rollon", {"name": "Neem Oil Roll-On", "description": ""}) == "30roll"
    assert family_rule("phyto-rollon", {"name": "Phytolacca Americana Oil", "description": "Apply with roll-on applicator"}) == "30roll"
    assert family_rule("rollon-x", {"name": "Herbal Rollon", "description": ""}) == "30roll"
    assert family_rule("roll-on-x", {"name": "Roll On Pain Relief", "description": ""}) == "30roll"
    # Non-roll-on should not match
    assert family_rule("plain", {"name": "Zinc Synergy", "description": ""}) is None


def test_synergy_syntropy_alias_fmp_join():
    """Storefront 'MSM Synergy' should join FMP row keyed as 'MSM Syntropy Powder'.

    The alias maps syntropy↔synergy so both sides normalize to the same token.
    The FMP index stores both the full key ('msm synergy powder') AND the
    suffix-stripped key ('msm synergy'), so the storefront lookup finds it.
    """
    from scripts.populate_bottle_types import _build_fmp_index

    # FMP row named 'MSM Syntropy Powder' with 45g powder packaging -> 30g
    fmp_rows = [
        {"product_name": "MSM Syntropy Powder",
         "zc_sold_display": "45g", "sold_measurement": "g", "type": "Functional Formulation"},
    ]
    fmp_by_name = _build_fmp_index(fmp_rows)

    products = {
        "msm-syntropy": {"name": "MSM Synergy"},
    }
    m = build_assignments(products, fmp_by_name)
    assert m["assignments"].get("msm-syntropy") == "30 g", (
        f"Expected 30g, got {m['assignments'].get('msm-syntropy')!r}; "
        f"review={m['review']}"
    )


def test_suffix_strip_powder_join():
    """Storefront 'Quercetin Dihydrate Powder' should join FMP 'Quercetin Dihydrate Powder\\n60 grams'.

    The FMP name has a multi-line entry; after normalization the FMP key becomes
    'quercetin dihydrate powder 60 grams'.  The suffix-stripped key (stripping
    the trailing word 'grams' is not in scope — only powder/tablets/capsules).
    Actually this tests the direct storefront-name-suffix-strip path:
    storefront 'Quercetin Dihydrate Powder' -> stripped -> 'Quercetin Dihydrate'
    which should match FMP 'Quercetin Dihydrate'.
    """
    from scripts.populate_bottle_types import _build_fmp_index

    # Simulate the FMP row (name without multi-line suffix in this unit test)
    fmp_rows = [
        {"product_name": "Quercetin Dihydrate",
         "zc_sold_display": "60g", "sold_measurement": "g", "type": "Pure Powders"},
    ]
    fmp_by_name = _build_fmp_index(fmp_rows)

    products = {
        "quercetin-dihydrate-powder": {"name": "Quercetin Dihydrate Powder"},
    }
    m = build_assignments(products, fmp_by_name)
    # Pure Powders + g -> 120cap
    assert m["assignments"].get("quercetin-dihydrate-powder") == "120 caps", (
        f"Expected 120cap, got {m['assignments'].get('quercetin-dihydrate-powder')!r}"
    )


def test_fuzzy_accepts_close_name():
    """A storefront name ≥0.92 similar to an FMP key → assigned if FMP row classifiable."""
    from scripts.populate_bottle_types import _build_fmp_index

    # 'glutathione synergy' vs 'glutathione syntropy' after alias both become
    # 'glutathione synergy', so that's an alias match not fuzzy.
    # Use a slightly misspelled name to test pure fuzzy: 'Zinc Synrgy' vs 'Zinc Synergy'
    # ('Zinc Synrgy' normalises to 'zinc synrgy', ratio=0.957 ≥ 0.92)
    fmp_rows = [
        {"product_name": "Zinc Synergy",
         "zc_sold_display": "30pullulan", "sold_measurement": "pullulan", "type": ""},
    ]
    fmp_by_name = _build_fmp_index(fmp_rows)

    products = {
        "zinc-synrgy": {"name": "Zinc Synrgy"},  # typo: missing 'e', ratio=0.957
    }
    m = build_assignments(products, fmp_by_name)
    assert m["assignments"].get("zinc-synrgy") == "30 Caps", (
        f"Expected 30cap from fuzzy, got {m['assignments'].get('zinc-synrgy')!r}"
    )


def test_fuzzy_rejects_too_different_name():
    """A name that is too different (<0.92 similarity) stays in review."""
    from scripts.populate_bottle_types import _build_fmp_index

    fmp_rows = [
        {"product_name": "Zinc Synergy",
         "zc_sold_display": "30pullulan", "sold_measurement": "pullulan", "type": ""},
    ]
    fmp_by_name = _build_fmp_index(fmp_rows)

    products = {
        "mystery-tonic": {"name": "Mystery Tonic"},
    }
    m = build_assignments(products, fmp_by_name)
    assert "mystery-tonic" not in m["assignments"]
    assert any(r["slug"] == "mystery-tonic" for r in m["review"])


def test_fuzzy_unclassifiable_fmp_row_goes_to_review():
    """Fuzzy matches an FMP row, but that row's packaging is unclassifiable → review."""
    from scripts.populate_bottle_types import _build_fmp_index

    # 30ml liquid → classify_from_fmp returns None (not in the 5/15/50/100ml set)
    fmp_rows = [
        {"product_name": "Zinc Synergy",
         "zc_sold_display": "30ml", "sold_measurement": "ml", "type": ""},
    ]
    fmp_by_name = _build_fmp_index(fmp_rows)

    products = {
        "zinc-synrgy": {"name": "Zinc Synrgy"},  # close enough for fuzzy (ratio=0.957)
    }
    m = build_assignments(products, fmp_by_name)
    assert "zinc-synrgy" not in m["assignments"], (
        "Should be in review because FMP row has unclassifiable packaging (30ml)"
    )
    assert any(r["slug"] == "zinc-synrgy" for r in m["review"])
