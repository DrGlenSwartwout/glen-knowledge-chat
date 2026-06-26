from dashboard.biofield_profile import mine_profile_stresses


def _extract(text):
    # stub: pretend the LLM pulled two labels from the free text when present
    return ["Chronic fatigue", "Poor sleep"] if text.strip() else []


def test_discrete_fields_list_and_string_forms():
    profile = {"tags": ["Inflammation", "Heavy metals"],
               "conditions": "Hashimoto's; Eczema",
               "terrain_concerns": "Acidic",
               "body_systems": ["Liver"]}
    out = mine_profile_stresses(profile, lambda t: [])
    assert set(out) == {"Inflammation", "Heavy metals", "Hashimoto's", "Eczema", "Acidic", "Liver"}


def test_free_text_goes_through_extract_and_dedupes():
    profile = {"tags": ["Inflammation"], "challenges": "always tired", "goals": "sleep better"}
    out = mine_profile_stresses(profile, _extract)
    assert "Inflammation" in out and "Chronic fatigue" in out and "Poor sleep" in out


def test_dedupe_case_insensitive():
    profile = {"tags": ["Inflammation", "inflammation"], "conditions": "INFLAMMATION"}
    out = mine_profile_stresses(profile, lambda t: [])
    assert out == ["Inflammation"]


def test_empty_profile():
    assert mine_profile_stresses({}, _extract) == []
    assert mine_profile_stresses(None, _extract) == []
