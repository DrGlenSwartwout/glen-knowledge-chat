from dashboard import biofield_portal_publish as bpp

def test_dosing_joins_present_fields_and_skips_blanks():
    assert bpp._dosing({"dosage": "1 capsule", "frequency": "daily",
                        "timing": "with food"}) == "1 capsule daily with food"
    assert bpp._dosing({"dosage": "10 drops", "frequency": "", "timing": ""}) == "10 drops"
    assert bpp._dosing({"dosage": "", "frequency": "", "timing": ""}) == ""

def test_segment_narrative_splits_layer_by_layer():
    layers = [{"remedy": "Vitality", "head": "ED3 Cell Driver"},
              {"remedy": "Chelation", "head": "Kidney"},
              {"remedy": "Nous Energy", "head": "Kidney"}]
    narr = ("Aloha Karin. The surface layer needs Vitality to restore energy. "
            "Next, Chelation clears the burden. Finally, Nous Energy steadies the mind.")
    segs = bpp.segment_narrative(narr, layers)
    assert len(segs) == 3
    assert "Vitality" in segs[0]
    assert "Chelation" in segs[1]
    assert "Nous Energy" in segs[2]

def test_segment_narrative_returns_empty_when_not_alignable():
    layers = [{"remedy": "Vitality", "head": "ED3"},
              {"remedy": "Chelation", "head": "Kidney"}]
    # Narrative never mentions the second remedy/head -> cannot align 1:1.
    assert bpp.segment_narrative("A generic message with no cues at all.", layers) == []
    assert bpp.segment_narrative("", layers) == []
