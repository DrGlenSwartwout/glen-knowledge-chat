import sqlite3
import pytest
from dashboard import glossary_crosslinks as gx


def test_canon_normalises_gland_and_plural():
    assert gx.canon("Adrenal Gland") == "adrenal"
    assert gx.canon("Bile Ducts") == "bile duct"
    assert gx.canon("Heart") == "heart"
    assert gx.canon("Muscle") == "muscle"          # not emptied


@pytest.fixture
def cx():
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    c.executescript(
        "CREATE TABLE e4l_items (code TEXT PRIMARY KEY, category TEXT, subcategory TEXT, "
        " name TEXT, full_name TEXT, e4l_description TEXT, clinical_notes TEXT, sort_order INTEGER);"
        "CREATE TABLE e4l_pattern_structures (code TEXT, structure TEXT, stype TEXT, is_primary INTEGER, source_phrase TEXT, PRIMARY KEY(code,structure));"
    )
    c.executemany("INSERT INTO e4l_items VALUES (?,?,?,?,?,?,?,?)", [
        ("ED1", "ED", "", "Source", "Source Driver", "d", "", 1),
        ("ED10", "ED", "", "Skin", "Skin Field", "d", "", 2),
    ])
    c.executemany("INSERT INTO e4l_pattern_structures VALUES (?,?,?,?,?)", [
        ("ED1", "Heart", "organ", 1, ""),
        ("ED1", "Adrenal Gland", "organ", 0, ""),
        ("ED10", "Heart", "organ", 0, ""),
        ("ED10", "Energy", "function", 0, ""),   # non-organ ignored
    ])
    c.commit()
    return c


def test_organ_to_patterns_groups_by_canon(cx):
    m = gx.organ_to_patterns(cx)
    # Heart is on both patterns
    heart = {p["slug"] for p in m["heart"]}
    assert heart == {"ed1", "ed10"}
    # Adrenal Gland canon -> "adrenal", only ED1
    assert [p["slug"] for p in m["adrenal"]] == ["ed1"]
    assert "energy" not in m               # function stype excluded


def test_clinical_organ_index():
    catalog = {"dimensions": [
        {"key": "organs", "entries": [{"slug": "adrenal", "name": "Adrenal"},
                                       {"slug": "heart", "name": "Heart"}]},
        {"key": "miasms", "entries": [{"slug": "x", "name": "X"}]},
    ]}
    idx = gx.clinical_organ_index(catalog)
    assert idx["adrenal"] == "adrenal"     # so "Adrenal Gland" (canon adrenal) links here
    assert idx["heart"] == "heart"
    assert "x" not in idx.values()          # only organs dimension
