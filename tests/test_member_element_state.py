import sqlite3
from dashboard import member_element_state as mes


def _cx():
    cx = sqlite3.connect(":memory:")
    mes.init_table(cx)
    return cx


def test_deficient_is_lowest_scoring_element():
    scores = {"Wood": 80, "Fire": 60, "Earth": 40, "Metal": 20, "Water": 5}
    assert mes.deficient_element(scores) == "Water"
    assert mes.dominant_element(scores) == "Wood"


def test_deficient_handles_empty_or_garbage():
    assert mes.deficient_element({}) is None
    assert mes.deficient_element(None) is None
    assert mes.deficient_element({"Wood": "n/a"}) is None


def test_upsert_then_get_roundtrips_and_derives():
    cx = _cx()
    scores = {"Wood": 80, "Fire": 60, "Earth": 40, "Metal": 20, "Water": 5}
    row = mes.upsert(cx, "Jane@Example.com ", scores, source="portal_chat")
    assert row["deficient_element"] == "Water"
    got = mes.get(cx, "jane@example.com")
    assert got["element_scores"]["Fire"] == 60
    assert got["dominant_element"] == "Wood"
    assert got["source"] == "portal_chat"


def test_upsert_overwrites_same_email():
    cx = _cx()
    mes.upsert(cx, "j@x.com", {"Wood": 80, "Fire": 1, "Earth": 40, "Metal": 20, "Water": 50})
    mes.upsert(cx, "j@x.com", {"Wood": 1, "Fire": 80, "Earth": 40, "Metal": 20, "Water": 50})
    got = mes.get(cx, "j@x.com")
    assert got["deficient_element"] == "Wood"
    assert cx.execute("SELECT COUNT(*) FROM member_element_state").fetchone()[0] == 1


def test_get_missing_returns_none():
    assert mes.get(_cx(), "nobody@x.com") is None
