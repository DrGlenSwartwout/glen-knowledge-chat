from dashboard import intake, intake_parse

FORM = intake.INTAKE_FORM


def test_prompt_includes_sections_and_scale_options():
    p = intake_parse.build_parse_prompt(FORM, "PASTED CLIENT TEXT")
    assert "PASTED CLIENT TEXT" in p
    assert "terrain" in p and "health_concerns" in p
    # a scale option label appears so the model maps to the integer
    assert "Rapid Aging" in p


def test_coerce_drops_unknown_and_bad_scale():
    raw = {
        "first_name": "Steven", "bogus_field": "x",
        "terrain": "2", "penetration": 99,  # 2 valid (coerced to int), 99 out of range -> dropped
        "commitment": 8,
        "health_concerns": [{"concern": "cataracts", "rating": 10, "junk": "drop"},
                            "not-a-row"],
        "terms": {"agreed": True},  # consent skipped
    }
    out = intake_parse.coerce_parsed(FORM, raw)
    assert out["first_name"] == "Steven"
    assert "bogus_field" not in out
    assert out["terrain"] == 2 and "penetration" not in out
    assert out["commitment"] == 8
    assert out["health_concerns"] == [{"concern": "cataracts", "rating": 10}]
    assert "terms" not in out


def test_parse_uses_complete_and_coerces():
    canned = '{"first_name":"Ann","terrain":3,"nope":1}'
    out = intake_parse.parse(FORM, "text", lambda system, user: canned)
    assert out == {"first_name": "Ann", "terrain": 3}


def test_parse_bad_json_returns_empty():
    out = intake_parse.parse(FORM, "text", lambda s, u: "not json")
    assert out == {}


def test_coerce_flattens_section_nested_output():
    """The LLM returns answers grouped by section; coerce must flatten one level."""
    raw = {
        "personal": {"first_name": "Steven", "last_name": "Fox"},
        "dimensions": {"terrain": 3, "penetration": 5, "commitment": 10},
        "goals": {"health_concerns": [{"concern": "cataracts", "rating": "10"}]},
    }
    out = intake_parse.coerce_parsed(FORM, raw)
    assert out["first_name"] == "Steven"
    assert out["terrain"] == 3 and out["penetration"] == 5 and out["commitment"] == 10
    assert out["health_concerns"] == [{"concern": "cataracts", "rating": "10"}]
