import json
from dashboard import clinical_glossary as cg


def _write(tmp_path):
    p = tmp_path / "cat.json"
    p.write_text(json.dumps({
        "dimensions": [
            {"key": "organs", "title": "Organs", "blurb": "b", "entry_count": 2,
             "entries": [
                 {"slug": "adrenal", "name": "Adrenal", "description": "d", "remedies": [{"name": "R", "url": "u"}]},
                 {"slug": "anus", "name": "Anus", "description": "d2", "remedies": []}]},
            {"key": "miasms", "title": "Miasms", "blurb": "b2", "entry_count": 1,
             "entries": [{"slug": "arsenicum", "name": "Arsenicum", "description": "d3", "remedies": []}]},
        ]
    }), encoding="utf-8")
    return str(p)


def test_load_and_dimensions(tmp_path):
    cat = cg.load(_write(tmp_path))
    dims = cg.dimensions(cat)
    assert [d["key"] for d in dims] == ["organs", "miasms"]
    assert dims[0]["entry_count"] == 2
    assert "entries" not in dims[0]  # hub list is lightweight


def test_get_dimension(tmp_path):
    cat = cg.load(_write(tmp_path))
    organs = cg.get_dimension("organs", cat)
    assert organs["title"] == "Organs"
    assert [e["slug"] for e in organs["entries"]] == ["adrenal", "anus"]
    assert cg.get_dimension("nope", cat) is None


def test_missing_file_is_empty():
    cat = cg.load("/no/such/file.json")
    assert cat == {"dimensions": []}
    assert cg.dimensions(cat) == []
    assert cg.get_dimension("organs", cat) is None
