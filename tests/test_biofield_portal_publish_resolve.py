from dashboard import biofield_portal_publish as bpp

CATALOG = {
    "vitality":       {"name": "Vitality"},
    "chelation":      {"name": "Chelation"},
    "nous-energy":    {"name": "Nous Energy"},
    "neuro-magnesium":{"name": "Neuro Magnesium"},
    "terrain-restore":{"name": "Terrain Restore"},
}

def test_alias_overrides_take_precedence():
    assert bpp.resolve_remedy_slug("Focus, Neuromagnesium", CATALOG) == "neuro-magnesium"
    assert bpp.resolve_remedy_slug("Focus Neuro-Magnesium", CATALOG) == "neuro-magnesium"
    assert bpp.resolve_remedy_slug(
        "Community Spirit Formula in Terrain Restore", CATALOG) == "terrain-restore"

def test_exact_names_resolve_via_name_to_slug():
    assert bpp.resolve_remedy_slug("Vitality", CATALOG) == "vitality"
    assert bpp.resolve_remedy_slug("Chelation", CATALOG) == "chelation"
    assert bpp.resolve_remedy_slug("Nous Energy", CATALOG) == "nous-energy"

def test_unresolvable_returns_none():
    assert bpp.resolve_remedy_slug("Totally Invented Remedy XYZ", CATALOG) is None
    assert bpp.resolve_remedy_slug("", CATALOG) is None
