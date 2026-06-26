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


def test_build_panel_skips_packaging_keeps_ingredients():
    """Packaging lines (ea. unit or packaging-word names) must be stripped; real
    ingredients must survive; panel must be generated (not held for review)."""
    items = [
        # real ingredient — must appear in output
        {"ingredient_canonical": "R-Lipoic Acid", "ingredient_name": None, "dose": 100, "dose_unit": "mg"},
        # blank-name capsule/bottle count (1 ea.) — packaging, must be skipped
        {"ingredient_canonical": None, "ingredient_name": None, "dose": 1, "dose_unit": "ea."},
        # named packaging component (Plantcaps®) — must be filtered by name
        {"ingredient_canonical": "Plantcaps®", "ingredient_name": None, "dose": 1, "dose_unit": "ea."},
    ]
    out, reason = build_panel(items)
    assert reason is None, f"Expected panel to be generated but got reason: {reason!r}"
    assert out == [{"name": "R-Lipoic Acid", "dose": "100 mg"}]
