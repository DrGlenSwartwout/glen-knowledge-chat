from scripts.populate_bottle_types import classify_from_fmp, family_rule, build_assignments

def test_classify_liquid_and_caps_and_powder():
    assert classify_from_fmp({"zc_sold_display": "50ml", "sold_measurement": "ml", "type": ""}) == "50ml"
    assert classify_from_fmp({"zc_sold_display": "30pullulan", "sold_measurement": "pullulan", "type": ""}) == "30cap"
    assert classify_from_fmp({"zc_sold_display": "120vegicaps", "sold_measurement": "vegicaps", "type": ""}) == "120cap"
    assert classify_from_fmp({"zc_sold_display": "30g", "sold_measurement": "g", "type": "Pure Powders"}) == "120cap"
    assert classify_from_fmp({"zc_sold_display": "30g", "sold_measurement": "g", "type": "Functional Formulation"}) == "30g"
    assert classify_from_fmp({"zc_sold_display": "1000ml", "sold_measurement": "ml", "type": ""}) is None  # bulk -> review

def test_family_rule_infoceutical_and_eyedrops():
    assert family_rule("ei8-x", {"name": "EI8 Microbes", "source": "infoceutical-catalog"}) == "30ml"
    assert family_rule("mb1-x", {"name": "MB1 Brain Stem Hologram"}) == "30ml"
    assert family_rule("drops", {"name": "ACES Eyedrops"}) == "5ml"
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
    assert m["assignments"]["cap"] == "30cap"    # fmp join
    assert "already" not in m["assignments"]      # never overwrite
    assert any(r["slug"] == "mystery" for r in m["review"])  # unmatched -> review
