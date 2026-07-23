from dashboard import oasis_roadmap as orm

def test_hero_tools_lead_and_ownership_excludes():
    rm = orm.build_roadmap(owned_slugs=set(), terrain_phase="cleanse")
    slugs = [x["slug"] for x in rm]
    assert slugs[:3] == ["harmony", "water-ionizer", "kloud"]     # hero order, above terrain items
    assert all(x["tier"] in ("hero", "terrain", "general") for x in rm)

def test_owned_hero_is_dropped_but_order_preserved():
    rm = orm.build_roadmap(owned_slugs={"water-ionizer"}, terrain_phase=None)
    slugs = [x["slug"] for x in rm]
    assert "water-ionizer" not in slugs
    assert slugs[:2] == ["harmony", "kloud"]                       # remaining heroes keep order

def test_terrain_phase_gap_items_follow_heroes():
    rm = orm.build_roadmap(owned_slugs={"harmony","water-ionizer","kloud"}, terrain_phase="cleanse")
    assert rm and all(x["tier"] != "hero" for x in rm)             # heroes owned -> only gap/general
