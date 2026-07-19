from dashboard.portal_view import _reveal_as_report_content


def test_layers_map_greeting_title_meaning_and_remedy_strings():
    reveal = {
        "interpretation": {"greeting": "Aloha", "body": "..."},
        "layers": [{"n": 1, "title": "Layer One", "meaning": "m1",
                    "remedy": {"name": "Calm Formula", "dosing": "2/day"}}],
        "remedies": [],
    }
    c = _reveal_as_report_content(reveal)
    assert c["greeting"] == "Aloha"
    assert c["layers"] == [{"n": 1, "title": "Layer One", "meaning": "m1",
                            "remedy": "Calm Formula", "dosing": "2/day"}]


def test_flat_reveal_without_layers_maps_remedies_to_layers():
    reveal = {"interpretation": {"greeting": "Hi"}, "layers": [],
              "remedies": [{"name": "Top Match", "meaning": "why", "dosing": "1/day"}]}
    c = _reveal_as_report_content(reveal)
    assert c["layers"] == [{"n": 1, "title": "", "meaning": "why",
                            "remedy": "Top Match", "dosing": "1/day"}]


def test_empty_reveal_yields_no_layers():
    assert _reveal_as_report_content({"interpretation": {}, "layers": [], "remedies": []})["layers"] == []
