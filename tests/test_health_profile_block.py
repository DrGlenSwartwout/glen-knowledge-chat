from dashboard import health_profile
import sqlite3

def test_editable_ids_include_dimensions_exclude_consent():
    ids = health_profile.EDITABLE_FIELD_IDS
    for included in ("terrain","penetration","tissue_layer","response","commitment","health_concerns"):
        assert included in ids        # dimensions are self-reported and change with healing -> editable
    assert "terms" not in ids         # consent/signature excluded

def test_build_block_off_when_disabled():
    assert health_profile.build_block(None, "a@b.com", False) == {"enabled": False}


def test_build_block_carries_dominant_terrain_options():
    cx = sqlite3.connect(":memory:")
    block = health_profile.build_block(cx, "a@b.com", True)
    terrain = next(
        field
        for section in block["sections"]
        for field in section["fields"]
        if field["id"] == "terrain"
    )
    assert terrain["options"] == [
        {"value": 1, "label": "Cancer, Degeneration, Viral or Low Energy"},
        {"value": 2, "label": "Rapid Aging, Bacterial, or Parasitic"},
        {"value": 3, "label": "Fungal, Deposition, Slow Metabolism, or Low Body Temperature"},
        {"value": 4, "label": "Allergy or Toxicity"},
        {"value": 5, "label": "Stress or Hormonal Imbalance"},
    ]
