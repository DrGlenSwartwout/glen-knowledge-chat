import sqlite3
from dashboard import recommendation_prefs as rp


def _cx():
    cx = sqlite3.connect(":memory:")
    rp.init_recommendation_prefs(cx)
    return cx


def test_operator_and_client_notes_independent():
    cx = _cx()
    rp.set_operator_note(cx, "A@B.com", "neuro-magnesium", "take at night")
    rp.set_client_note(cx, "a@b.com", "neuro-magnesium", "helped my sleep")
    n = rp.get_notes(cx, "a@b.com")["neuro-magnesium"]
    assert n["operator_note"] == "take at night"
    assert n["client_note"] == "helped my sleep"
    # updating one preserves the other
    rp.set_operator_note(cx, "a@b.com", "neuro-magnesium", "morning now")
    n = rp.get_notes(cx, "a@b.com")["neuro-magnesium"]
    assert n["operator_note"] == "morning now" and n["client_note"] == "helped my sleep"


def test_notes_empty_and_blank_guards():
    cx = _cx()
    assert rp.get_notes(cx, "nobody@x.com") == {}
    rp.set_client_note(cx, "", "x", "n")          # blank email -> no-op
    rp.set_client_note(cx, "a@b.com", "", "n")    # blank product -> no-op
    assert rp.get_notes(cx, "a@b.com") == {}


def test_section_state_roundtrip():
    cx = _cx()
    assert rp.get_section_state(cx, "a@b.com") == {}
    rp.set_section_state(cx, "A@B.com", "biofield", True)
    rp.set_section_state(cx, "a@b.com", "purchased", False)
    assert rp.get_section_state(cx, "a@b.com") == {"biofield": True, "purchased": False}
    rp.set_section_state(cx, "a@b.com", "biofield", False)   # toggle
    assert rp.get_section_state(cx, "a@b.com")["biofield"] is False
