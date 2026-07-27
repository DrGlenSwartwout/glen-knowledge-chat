import json

import bodymap_store
from dashboard import bodymap_homework


def _first_zone(system):
    return bodymap_store.zone_ids(system)[0]


def test_valid_payload_normalizes():
    zone = _first_zone("organs")
    payload = json.dumps({
        "system": "organs",
        "marks": [{"zone": zone, "note": "  a bit tender  "}],
        "note": "  overall reflection  ",
    })
    parsed = bodymap_homework.parse_marks(payload)
    assert parsed is not None
    assert parsed["system"] == "organs"
    assert len(parsed["marks"]) == 1
    mark = parsed["marks"][0]
    assert mark["zone"] == zone
    assert mark["note"] == "a bit tender"
    assert mark["anatomy"]  # filled from bodymap_store
    assert parsed["note"] == "overall reflection"


def test_unknown_system_returns_none():
    payload = json.dumps({"system": "nope", "marks": []})
    assert bodymap_homework.parse_marks(payload) is None


def test_real_but_disallowed_system_returns_none():
    assert "iridology" in bodymap_store.SYSTEMS
    assert "iridology" not in bodymap_homework.WHOLE_BODY_SYSTEMS
    payload = json.dumps({"system": "iridology", "marks": []})
    assert bodymap_homework.parse_marks(payload) is None


def test_invalid_zone_id_dropped():
    payload = json.dumps({
        "system": "organs",
        "marks": [{"zone": "not-a-real-zone", "note": "x"}],
    })
    parsed = bodymap_homework.parse_marks(payload)
    assert parsed is not None
    assert parsed["marks"] == []


def test_non_json_string_returns_none():
    assert bodymap_homework.parse_marks("just reflective text") is None


def test_note_length_capped():
    zone = _first_zone("organs")
    payload = json.dumps({
        "system": "organs",
        "marks": [{"zone": zone, "note": "x" * 999}],
        "note": "y" * 9999,
    })
    parsed = bodymap_homework.parse_marks(payload)
    assert parsed is not None
    assert len(parsed["marks"][0]["note"]) == 500
    assert len(parsed["note"]) == 2000


def test_summarize_marks_valid_payload():
    zone = _first_zone("organs")
    payload = json.dumps({
        "system": "organs",
        "marks": [{"zone": zone, "note": "tender"}],
        "note": "feeling okay overall",
    })
    summary = bodymap_homework.summarize_marks(payload)
    assert summary is not None
    assert "Organs" in summary
    assert "Areas of concern" in summary


def test_summarize_marks_free_text_returns_none():
    assert bodymap_homework.summarize_marks("just reflective text") is None


def test_has_content_marks_only():
    assert bodymap_homework.has_content({"marks": [{"zone": "organ-brain"}], "note": ""}) is True


def test_has_content_note_only():
    assert bodymap_homework.has_content({"marks": [], "note": "something"}) is True


def test_has_content_both_empty():
    assert bodymap_homework.has_content({"marks": [], "note": ""}) is False
    assert bodymap_homework.has_content({"marks": [], "note": "   "}) is False
