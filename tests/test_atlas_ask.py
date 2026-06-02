import atlas_ask


def test_match_concepts_ranks_by_keyword_overlap():
    concepts = [
        {"id": "light-therapy", "label": "Light Therapy",
         "aliases": ["syntonics"], "summary": "light frequencies for vision"},
        {"id": "detox", "label": "Detox", "aliases": [], "summary": "elimination"},
    ]
    ids = atlas_ask.match_concepts("what light helps my vision?", concepts, k=1)
    assert ids == ["light-therapy"]


def test_match_concepts_uses_aliases():
    concepts = [{"id": "biofield", "label": "Biofield",
                 "aliases": ["energy field"], "summary": "organizing field"}]
    ids = atlas_ask.match_concepts("tell me about the energy field", concepts, k=1)
    assert ids == ["biofield"]
