from dashboard import oasis_roadmap as orm

_HERO_ORDER = ["harmony-laser", "water-ionizer-15plate", "kloud-pemf-maxi"]


def test_hero_tools_lead_in_fixed_order():
    rm = orm.build_roadmap(owned_slugs=set())
    slugs = [x["slug"] for x in rm]
    assert slugs[:3] == _HERO_ORDER                # hero order, above secondary items
    assert all(x["tier"] in ("hero", "secondary") for x in rm)


def test_secondary_tools_follow_heroes_and_all_present():
    rm = orm.build_roadmap(owned_slugs=set())
    secondary_slugs = [x["slug"] for x in rm if x["tier"] == "secondary"]
    assert len(secondary_slugs) == len(orm.SECONDARY_TOOLS)
    assert secondary_slugs == [t["slug"] for t in orm.SECONDARY_TOOLS]  # list order preserved


def test_owned_hero_is_dropped_but_order_preserved():
    rm = orm.build_roadmap(owned_slugs={"water-ionizer-15plate"})
    slugs = [x["slug"] for x in rm]
    assert "water-ionizer-15plate" not in slugs
    assert slugs[:2] == ["harmony-laser", "kloud-pemf-maxi"]    # remaining heroes keep order


def test_owned_secondary_is_excluded():
    rm = orm.build_roadmap(owned_slugs={"dowsing-rods"})
    slugs = [x["slug"] for x in rm]
    assert "dowsing-rods" not in slugs


def test_all_heroes_owned_leaves_only_secondary():
    rm = orm.build_roadmap(owned_slugs=set(_HERO_ORDER))
    assert rm and all(x["tier"] != "hero" for x in rm)


def test_terrain_phase_argument_is_ignored():
    # Healing tools work in all phases now -- terrain_phase must not change
    # the tool set or its order, whether it's a known phase, an unknown
    # string, or None.
    base = orm.build_roadmap(owned_slugs=set())
    for phase in ("cleanse", "energize", "not-a-real-phase", None):
        assert orm.build_roadmap(owned_slugs=set(), terrain_phase=phase) == base


def test_secondary_items_have_empty_why():
    rm = orm.build_roadmap(owned_slugs=set())
    secondary = [x for x in rm if x["tier"] == "secondary"]
    assert secondary and all(x["why"] == "" for x in secondary)
