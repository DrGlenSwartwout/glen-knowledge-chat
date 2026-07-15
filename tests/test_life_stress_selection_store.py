import sqlite3
from dashboard import life_stress_selection as sel

def _cx():
    cx = sqlite3.connect(":memory:")
    sel.init_table(cx)
    return cx

def test_set_get_roundtrip():
    cx = _cx()
    sel.set(cx, "a@b.com", ["mimulus-...", "aspen-..."])
    assert sel.get(cx, "a@b.com") == ["mimulus-...", "aspen-..."]

def test_get_missing_is_empty():
    assert sel.get(_cx(), "nobody@x.com") == []

def test_set_overwrites():
    cx = _cx()
    sel.set(cx, "a@b.com", ["x"]); sel.set(cx, "a@b.com", ["y", "z"])
    assert sel.get(cx, "a@b.com") == ["y", "z"]

def test_clear():
    cx = _cx()
    sel.set(cx, "a@b.com", ["x"]); sel.clear(cx, "a@b.com")
    assert sel.get(cx, "a@b.com") == []

def test_bad_json_returns_empty():
    cx = _cx()
    cx.execute("INSERT INTO life_stress_selections(email,slugs_json,updated_at) VALUES(?,?,?)",
               ("a@b.com", "{not json", "2026-07-14"))
    assert sel.get(cx, "a@b.com") == []
