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


def test_product_name_index_and_exact_resolve():
    products = {"liver-support": {"name": "Liver Support"},
                "flow-ease": {"name": "Flow Ease"}}
    idx = cg.product_name_index(products)
    assert cg.remedy_product_slug("Liver Support", idx) == "liver-support"
    assert cg.remedy_product_slug("flow  ease", idx) == "flow-ease"   # normalised
    assert cg.remedy_product_slug("Nonexistent", idx) is None
    assert cg.remedy_product_slug(".)", idx) is None                  # noise -> None


def test_product_name_index_slug_fallback():
    # product display name differs from slug (Synergy->Syntropy rename); a remedy
    # named after the slug still resolves via the slug key.
    products = {"sleep-syntropy": {"name": "Sleep Synergy"}}
    idx = cg.product_name_index(products)
    assert cg.remedy_product_slug("Sleep Syntropy", idx) == "sleep-syntropy"   # slug key
    assert cg.remedy_product_slug("Sleep Synergy", idx) == "sleep-syntropy"    # name key
    # a real display name still wins over a slug key collision
    p2 = {"foo-bar": {"name": "Real Name"}, "real-name": {"name": "Other"}}
    idx2 = cg.product_name_index(p2)
    assert cg.remedy_product_slug("Real Name", idx2) == "foo-bar"


def test_override_resolves_non_exact():
    idx = {}
    ov = {"Bicarbonate Blend": "alkalize-bicarbonate-blend"}
    assert cg.remedy_product_slug("Bicarbonate Blend", idx, ov) == "alkalize-bicarbonate-blend"
    assert cg.remedy_product_slug("Unmapped", idx, ov) is None       # no fuzzy


def test_with_product_links_attaches_slug():
    dim = {"key": "organs", "entries": [
        {"slug": "adrenal", "name": "Adrenal",
         "remedies": [{"name": "Liver Support", "url": "u"}, {"name": "Mystery", "url": "u2"}]}]}
    idx = {"liver support": "liver-support"}
    out = cg.with_product_links(dim, idx)
    rem = out["entries"][0]["remedies"]
    assert rem[0]["product_slug"] == "liver-support"
    assert rem[1]["product_slug"] is None


def test_supplementary_environmental_dimension_merged():
    # load() with no path merges the curated e4l_stressor_map.json dimensions.
    cat = cg.load()
    env = cg.get_dimension("environmental", cat)
    assert env is not None and len(env["entries"]) >= 20
    lead = next(e for e in env["entries"] if e["name"] == "Lead")
    assert lead["affects"] and lead["remedies"]
    # an explicit path skips the merge (isolation for other tests)
    import json, tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.json")
        json.dump({"dimensions": [{"key": "x", "entries": []}]}, open(p, "w"))
        assert [d["key"] for d in cg.load(p)["dimensions"]] == ["x"]
