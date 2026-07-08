from dashboard.biofield_invoice import BIOFIELD_SLUG, resolve_line_slug, build_invoice_lines

CATALOG = [{"slug": "liver-support", "name": "Liver Support"},
           {"slug": "vitality", "name": "Vitality"},
           {"slug": "gastrozyme", "name": "GastroZyme"}]


def test_resolve_exact_case_insensitive():
    assert resolve_line_slug("liver support", CATALOG) == "liver-support"
    assert resolve_line_slug("GASTROZYME", CATALOG) == "gastrozyme"


def test_no_fuzzy_substitution():
    # a near miss must NOT silently resolve to a different SKU
    assert resolve_line_slug("Gastrozime", CATALOG) is None


def test_exact_wins_no_near_duplicate_jump():
    cat = [{"slug": "es1-lymph", "name": "ES1 Lymph"},
           {"slug": "es13-x", "name": "ES13 Something"}]
    assert resolve_line_slug("ES1 Lymph", cat) == "es1-lymph"     # exact wins
    assert resolve_line_slug("ES1", cat) is None                  # partial does NOT jump to ES13


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


def test_doses_per_day_parsing():
    from dashboard.biofield_invoice import doses_per_day
    assert doses_per_day("daily") == 1
    assert doses_per_day("a day") == 1
    assert doses_per_day("two times a day") == 2
    assert doses_per_day("twice a day") == 2
    assert doses_per_day("3 times a day") == 3
    assert doses_per_day("") is None and doses_per_day("as needed") is None


def test_bottles_needed():
    from dashboard.biofield_invoice import bottles_needed
    assert bottles_needed("two times a day", 30) == 2      # B17: ceil(2*30/30)
    assert bottles_needed("3 times a day", 100) == 1       # Green Jasper: ceil(90/100)
    assert bottles_needed("daily", 30) == 1
    assert bottles_needed("daily", "") == 1                # no doses_per_bottle (infoceutical) -> 1
    assert bottles_needed("gibberish", 30) == 1            # unparseable -> 1


def test_build_invoice_lines_honors_per_remedy_qty():
    cat = [{"slug": "b17-syntropy", "name": "B17 Syntropy"}]
    out = build_invoice_lines({}, [{"name": "B17 Syntropy", "qty": 2}], cat)
    b17 = [l for l in out["lines"] if l["slug"] == "b17-syntropy"][0]
    assert b17["qty"] == 2
    # backward-compat: a bare string still means qty 1
    out2 = build_invoice_lines({}, ["B17 Syntropy"], cat)
    assert [l for l in out2["lines"] if l["slug"] == "b17-syntropy"][0]["qty"] == 1
