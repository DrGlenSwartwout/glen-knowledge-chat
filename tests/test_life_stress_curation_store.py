import sqlite3
from dashboard import life_stress_curation as cur

def _cx():
    cx = sqlite3.connect(":memory:"); cur.init_table(cx); return cx

def test_set_get_roundtrip():
    cx = _cx(); cur.set(cx, "A@B.com", "42", ["x-in-terrain-restore"], "note here")
    g = cur.get(cx, "a@b.com")
    assert g["slugs"] == ["x-in-terrain-restore"] and g["note"] == "note here" and g["updated_at"]

def test_missing_is_none():
    assert cur.get(_cx(), "nobody@x.com") is None

def test_empty_slugs_is_none():
    cx = _cx(); cur.set(cx, "a@b.com", "42", [], "")
    assert cur.get(cx, "a@b.com") is None

def test_set_overwrites():
    cx = _cx(); cur.set(cx, "a@b.com", "42", ["x"], "n1"); cur.set(cx, "a@b.com", "42", ["y","z"], "n2")
    assert cur.get(cx, "a@b.com")["slugs"] == ["y","z"]

def test_clear():
    cx = _cx(); cur.set(cx, "a@b.com", "42", ["x"], ""); cur.clear(cx, "a@b.com")
    assert cur.get(cx, "a@b.com") is None

def test_bad_json_is_none():
    cx = _cx()
    cx.execute("INSERT INTO life_stress_curations(patient_email,practitioner_id,slugs_json,note,updated_at) VALUES(?,?,?,?,?)",
               ("a@b.com","42","{bad","","2026-07-14"))
    assert cur.get(cx, "a@b.com") is None
