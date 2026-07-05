import sqlite3
from dashboard import member_element_state as mes
from dashboard import portal_element_view as pev


def _cx():
    cx = sqlite3.connect(":memory:")
    mes.init_table(cx)
    return cx


def test_view_adds_lowercased_setting():
    cx = _cx()
    mes.upsert(cx, "j@x.com", {"Wood": 80, "Fire": 60, "Earth": 40, "Metal": 20, "Water": 5})
    view = pev.element_view(cx, "j@x.com")
    assert view["deficient_element"] == "Water"
    assert view["setting"] == "water"


def test_view_is_none_when_no_state():
    assert pev.element_view(_cx(), "nobody@x.com") is None


def test_view_surfaces_saved_override():
    cx = _cx()
    mes.upsert(cx, "j@x.com", {"Wood": 80, "Fire": 60, "Earth": 40, "Metal": 20, "Water": 5})
    mes.set_override(cx, "j@x.com", "fire")
    view = pev.element_view(cx, "j@x.com")
    assert view["setting"] == "water"       # computed, unchanged
    assert view["override"] == "fire"       # member's saved choice


def test_view_override_absent_is_none():
    cx = _cx()
    mes.upsert(cx, "j@x.com", {"Wood": 80, "Fire": 60, "Earth": 40, "Metal": 20, "Water": 5})
    assert pev.element_view(cx, "j@x.com")["override"] is None


def test_view_override_only_member_has_no_setting():
    cx = _cx()
    mes.set_override(cx, "j@x.com", "metal")   # chose a scene, no analysis yet
    view = pev.element_view(cx, "j@x.com")
    assert view["setting"] is None
    assert view["override"] == "metal"
