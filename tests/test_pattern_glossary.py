import sqlite3
import pytest
from dashboard import pattern_glossary as pg


def _seed(cx):
    cx.executescript(
        "CREATE TABLE e4l_items (code TEXT PRIMARY KEY, category TEXT NOT NULL, "
        " subcategory TEXT, name TEXT NOT NULL, full_name TEXT, e4l_description TEXT, "
        " clinical_notes TEXT, sort_order INTEGER);"
        "CREATE TABLE e4l_pattern_structures (code TEXT NOT NULL, structure TEXT NOT NULL, "
        " stype TEXT, is_primary INTEGER DEFAULT 0, source_phrase TEXT, PRIMARY KEY(code,structure));"
        "CREATE TABLE formulations (id INTEGER PRIMARY KEY, name TEXT NOT NULL, url TEXT);"
        "CREATE TABLE e4l_formulation_map (id INTEGER PRIMARY KEY, item_code TEXT, "
        " formulation_id INTEGER, priority INTEGER);"
    )
    cx.executemany("INSERT INTO formulations (id,name,url) VALUES (?,?,?)", [
        (1, "Nous Energy", ""), (2, "Holy Grail", ""), (3, "Terrain Restore", "")])
    cx.executemany("INSERT INTO e4l_formulation_map (item_code,formulation_id,priority) VALUES (?,?,?)", [
        ("ED1", 3, 3), ("ED1", 1, 1), ("ED1", 2, 2), ("ED1", 1, 4)])  # dup formulation 1
    cx.executemany("INSERT INTO e4l_items VALUES (?,?,?,?,?,?,?,?)", [
        ("ED1", "ED", "", "Source", "Source Driver", "Supports the body's fundamental energy source.", "", 1),
        ("Lead", "Environmental", "Heavy Metals", "Lead", "Lead", "", "", 2),   # structures-only
        ("ES9", "ES", "", "Ghost", "Ghost", "", "", 3),                          # neither -> excluded
    ])
    cx.executemany("INSERT INTO e4l_pattern_structures VALUES (?,?,?,?,?)", [
        ("ED1", "Heart", "organ", 1, ""),
        ("ED1", "Energy & Stamina", "function", 0, ""),
        ("Lead", "Nervous System", "system", 1, ""),
    ])
    cx.commit()


@pytest.fixture
def cx():
    c = sqlite3.connect(":memory:"); c.row_factory = sqlite3.Row
    _seed(c); return c


def test_slug_for_uses_code(cx):
    assert pg.slug_for("Heavy Metals") == "heavy-metals"
    assert pg.slug_for("ED1") == "ed1"


def test_pattern_remedies_ordered_and_deduped(cx):
    r = pg.pattern_remedies(cx, "ED1")
    # ordered by priority (1,2,3), formulation 1 deduped to its best (priority 1)
    assert [x["name"] for x in r] == ["Nous Energy", "Holy Grail", "Terrain Restore"]
    assert r[0]["priority"] == 1
    assert pg.pattern_remedies(cx, "NONE") == []
    assert pg.pattern_remedies(cx, "") == []


def test_get_pattern_shape_and_structure_order(cx):
    p = pg.get_pattern(cx, "ed1")
    assert p["code"] == "ED1" and p["name"] == "Source" and p["full_name"] == "Source Driver"
    assert p["description"].startswith("Supports the body")
    # primary structure first, then by stype/structure
    assert [s["structure"] for s in p["structures"]] == ["Heart", "Energy & Stamina"]
    assert p["structures"][0]["is_primary"] == 1
    assert p["has_page"] is True


def test_get_pattern_unknown_slug_is_none(cx):
    assert pg.get_pattern(cx, "nope") is None


def test_page_exists_rules(cx):
    assert pg.page_exists(cx, "ed1") is True          # described + structures
    assert pg.page_exists(cx, "lead") is True          # structures only
    assert pg.page_exists(cx, "ghost") is False        # neither
    assert pg.page_exists(cx, "missing") is False


def test_list_patterns_groups_and_excludes_empty(cx):
    groups = pg.list_patterns(cx)
    flat = {p["slug"]: p for g in groups for p in g["patterns"]}
    assert "ed1" in flat and "lead" in flat
    assert "ghost" not in flat                          # excluded (no page)
    assert flat["ed1"]["has_desc"] is True and flat["ed1"]["n_structures"] == 2
    assert flat["lead"]["has_desc"] is False and flat["lead"]["n_structures"] == 1
    cats = [g["category"] for g in groups]
    assert cats == sorted(set(cats), key=cats.index)   # each category once, stable
