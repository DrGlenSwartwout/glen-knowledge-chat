import json
import sqlite3
from dashboard.canonical_tags import get_person, init_tables, rebuild_people_columns, set_attr


def _cx(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "c.db"))
    init_tables(cx)
    cx.execute("CREATE TABLE people(email TEXT, tags TEXT DEFAULT '[]', conditions TEXT DEFAULT '[]', "
               "terrain_concerns TEXT DEFAULT '[]', body_systems TEXT DEFAULT '[]', "
               "challenges TEXT DEFAULT '', goals TEXT DEFAULT '')")
    cx.execute("INSERT INTO people(email) VALUES('j@x.com')")
    cx.commit()
    return cx


def test_get_person_reconstructs(tmp_path):
    cx = _cx(tmp_path)
    set_attr(cx, "j@x.com", "conditions", "Eczema", source="manual")
    set_attr(cx, "j@x.com", "conditions", "Asthma", source="manual")
    set_attr(cx, "j@x.com", "goals", "sleep better", source="manual")
    p = get_person(cx, "J@x.com")
    assert p["conditions"] == ["Asthma", "Eczema"]           # sorted
    assert p["goals"] == "sleep better"
    assert p["tags"] == [] and p["challenges"] == ""         # all keys present
    assert get_person(cx, "nobody@x.com")["conditions"] == []


def test_rebuild_writes_people_columns(tmp_path):
    cx = _cx(tmp_path)
    set_attr(cx, "j@x.com", "conditions", "Eczema", source="manual")
    set_attr(cx, "j@x.com", "body_systems", "Liver", source="manual")
    set_attr(cx, "j@x.com", "challenges", "always tired", source="manual")
    rebuild_people_columns(cx, "j@x.com")
    row = cx.execute("SELECT conditions, body_systems, challenges, tags FROM people "
                     "WHERE email='j@x.com'").fetchone()
    assert json.loads(row[0]) == ["Eczema"] and json.loads(row[1]) == ["Liver"]
    assert row[2] == "always tired" and json.loads(row[3]) == []
