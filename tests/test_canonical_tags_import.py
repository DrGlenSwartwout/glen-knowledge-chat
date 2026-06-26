"""Task 3 — import_from_people: seeds canonical store from people.* columns."""
import json
import sqlite3
from dashboard.canonical_tags import import_from_people, init_tables, get_person


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_tables(cx)
    cx.execute("CREATE TABLE people(email TEXT, tags TEXT, conditions TEXT, "
               "terrain_concerns TEXT, body_systems TEXT, challenges TEXT, goals TEXT)")
    cx.execute("INSERT INTO people VALUES(?,?,?,?,?,?,?)",
               ("j@x.com", json.dumps(["type:client", "Inflammation"]),
                json.dumps(["Eczema"]), "not json", "[]", "always tired", "more energy"))
    cx.execute("INSERT INTO people VALUES(?,?,?,?,?,?,?)",
               ("", "[]", "[]", "[]", "[]", "", ""))            # blank email skipped
    cx.commit()
    return cx


def test_import_seeds_store_from_people(tmp_path):
    cx = _cx(tmp_path)
    res = import_from_people(cx)
    assert res["persons"] == 1                                 # blank-email row skipped
    p = get_person(cx, "j@x.com")
    assert set(p["tags"]) == {"type:client", "Inflammation"}
    assert p["conditions"] == ["Eczema"] and p["terrain_concerns"] == []   # bad JSON -> []
    assert p["challenges"] == "always tired" and p["goals"] == "more energy"
    # all imported with source='import'
    srcs = {r[0] for r in cx.execute("SELECT DISTINCT source FROM person_attributes").fetchall()}
    assert srcs == {"import"}


def test_import_idempotent(tmp_path):
    cx = _cx(tmp_path)
    import_from_people(cx)
    n1 = cx.execute("SELECT COUNT(*) FROM person_attributes").fetchone()[0]
    import_from_people(cx)                                     # re-run
    n2 = cx.execute("SELECT COUNT(*) FROM person_attributes").fetchone()[0]
    assert n1 == n2 and n1 > 0
