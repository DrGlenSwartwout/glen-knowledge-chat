from scripts.generate_panels_from_db import build_panel


def test_build_panel_complete_and_incomplete():
    ok, reason = build_panel([
        {"ingredient_canonical": "R-Lipoic Acid", "ingredient_name": None, "dose": 100, "dose_unit": "mg"},
        {"ingredient_canonical": None, "ingredient_name": "Benfotiamine", "dose": 100, "dose_unit": "mg"},
    ])
    assert reason is None
    assert {"name": "R-Lipoic Acid", "dose": "100 mg"} in ok
    assert {"name": "Benfotiamine", "dose": "100 mg"} in ok
    bad, reason2 = build_panel([
        {"ingredient_canonical": None, "ingredient_name": None, "dose": 400, "dose_unit": "mg"},  # dosed, unnamed
    ])
    assert bad is None and "incomplete" in reason2.lower()
