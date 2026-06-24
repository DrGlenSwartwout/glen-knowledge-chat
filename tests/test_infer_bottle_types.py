from scripts.infer_bottle_types import infer_bottle_type, build_mapping

def test_dropper_15ml():
    t, c = infer_bottle_type({"name": "Foo 15 ml Dropper", "description": ""})
    assert t == "15ml" and c >= 0.9

def test_rollon():
    t, c = infer_bottle_type({"name": "Bar Roll-On 30 ml", "description": ""})
    assert t == "30roll" and c >= 0.7

def test_powder_cosmetic():
    t, _ = infer_bottle_type({"name": "Baz Powder 30 g", "description": ""})
    assert t == "100cos"

def test_low_confidence_defaults_and_flags_for_review():
    m = build_mapping({"x": {"name": "Mystery Tonic", "description": ""}})
    assert m["assignments"]["x"] == "default"
    assert any(r["slug"] == "x" for r in m["review"])

def test_existing_bottle_type_is_preserved():
    m = build_mapping({"y": {"name": "Foo 15 ml Dropper", "bottle_type": "5ml"}})
    assert "y" not in m["assignments"]  # already set -> not reassigned

def test_ambiguous_wide_mouth_goes_to_review():
    m = build_mapping({"z": {"name": "Wide Mouth 100 ml", "description": ""}})
    assert m["assignments"]["z"] == "default"
    assert any(r["slug"] == "z" for r in m["review"])
