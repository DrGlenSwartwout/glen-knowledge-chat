# tests/test_biofield_narrative_profile.py
from dashboard.biofield_narrative import build_narrative_prompt

_REP = {"client": {"name": "Jane"}, "date": "2026-06-25", "layers": []}


def test_profile_none_is_backcompat():
    a = build_narrative_prompt(_REP, "notes")
    b = build_narrative_prompt(_REP, "notes", profile=None)
    assert a == b
    assert "CLIENT-STATED" not in a["user"]


def test_profile_content_appended():
    prof = {"challenges": "always tired", "goals": "sleep better", "conditions": "Eczema"}
    p = build_narrative_prompt(_REP, "notes", profile=prof)
    assert "always tired" in p["user"] and "sleep better" in p["user"]
    assert "CLIENT-STATED" in p["user"]          # the profile block header
    assert p["system"] != build_narrative_prompt(_REP, "notes")["system"]   # guidance appended


def test_empty_profile_no_block():
    p = build_narrative_prompt(_REP, "notes", profile={})
    assert "CLIENT-STATED" not in p["user"]
    assert p["system"] == build_narrative_prompt(_REP, "notes")["system"]
