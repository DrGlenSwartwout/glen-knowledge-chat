import sqlite3
from dashboard import biofield_e4l as be


def _mk():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    cx.executescript(
        "CREATE TABLE e4l_items (code TEXT PRIMARY KEY, category TEXT, subcategory TEXT, "
        " name TEXT, full_name TEXT, e4l_description TEXT, clinical_notes TEXT, sort_order INTEGER);"
        "CREATE TABLE e4l_pattern_structures (code TEXT, structure TEXT, stype TEXT, is_primary INTEGER, source_phrase TEXT, PRIMARY KEY(code,structure));"
    )
    cx.execute("INSERT INTO e4l_items VALUES ('ED1','ED','','Source','Source Driver','A description.','',1)")
    cx.execute("INSERT INTO e4l_items VALUES ('ER5','ER','','Bare','Bare','','',2)")  # no desc, no struct
    cx.commit()
    return cx


def test_pattern_hrefs_link_only_pages():
    cx = _mk()
    hrefs = be._pattern_hrefs(cx, ["ED1", "ER5"])
    assert hrefs["ED1"] == "/learn/pattern/ed1"
    assert hrefs["ER5"] == ""     # no page (no description, no structures)
