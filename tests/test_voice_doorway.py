from dashboard.voice_doorway import voice_signal_tags


def test_voice_signal_tags_builds_prefixed_tags():
    tags = voice_signal_tags({
        "dominant_element": "Water",
        "dominant_treasure": "Jing",
        "polyvagal_state": {"ventral_vagal": 60, "sympathetic": 28, "dorsal_vagal": 12},
        "top_themes": ["grounding", "kidney-essence depletion"],
    })
    assert "element:water" in tags
    assert "treasure:jing" in tags
    assert "state:ventral-vagal" in tags
    assert "theme:grounding" in tags
    assert "theme:kidney-essence-depletion" in tags


def test_voice_signal_tags_empty_is_empty():
    assert voice_signal_tags({}) == []


def test_voice_signal_tags_polyvagal_string_ok():
    assert "state:sympathetic" in voice_signal_tags({"polyvagal_state": "sympathetic"})


def test_quiz_land_repointed_to_internal_doorway():
    import importlib, begin_funnel
    importlib.reload(begin_funnel)
    card = begin_funnel.CARD_CATALOG["quiz"]
    assert card["base_url"] == "/begin/doorway"
    assert card["internal"] is True
    assert begin_funnel.WANT_TARGETS["quiz"] == "/begin/doorway"
    # no scoreapp URL left in the funnel land config
    import json as _j
    assert "scoreapp.com" not in _j.dumps(begin_funnel.CARD_CATALOG["quiz"])
