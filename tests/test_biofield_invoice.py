from dashboard.biofield_invoice import BIOFIELD_SLUG, resolve_line_slug, build_invoice_lines

CATALOG = [{"slug": "liver-support", "name": "Liver Support"},
           {"slug": "vitality", "name": "Vitality"},
           {"slug": "gastrozyme", "name": "GastroZyme"}]


def test_resolve_exact_case_insensitive():
    assert resolve_line_slug("liver support", CATALOG) == "liver-support"
    assert resolve_line_slug("GASTROZYME", CATALOG) == "gastrozyme"


def test_resolve_fuzzy_close_match():
    # minor spelling drift still resolves (difflib cutoff 0.82)
    assert resolve_line_slug("Gastrozime", CATALOG) == "gastrozyme"


def test_resolve_none_when_no_match():
    assert resolve_line_slug("Green Jasper Gem Elixir", CATALOG) is None
    assert resolve_line_slug("", CATALOG) is None


def test_biofield_is_always_top_line():
    out = build_invoice_lines({"email": "d@x.com"}, ["Liver Support"], CATALOG)
    assert out["lines"][0] == {"slug": BIOFIELD_SLUG, "qty": 1}


def test_resolvable_remedies_become_lines_unresolvable_skipped():
    out = build_invoice_lines({"email": "d@x.com"},
                              ["Liver Support", "Vitality", "Green Jasper Gem Elixir"], CATALOG)
    slugs = [l["slug"] for l in out["lines"]]
    assert slugs == [BIOFIELD_SLUG, "liver-support", "vitality"]   # order preserved, biofield first
    assert out["skipped"] == ["Green Jasper Gem Elixir"]
    assert all(l["qty"] == 1 for l in out["lines"])
