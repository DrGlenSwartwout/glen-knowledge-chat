from dashboard import oasis_roadmap as orm

_HERO_ORDER = ["harmony-laser", "water-ionizer-15plate", "kloud-pemf-maxi"]

_REMOVED_SLUGS = ["smokey-quartz-healing-tool", "hypoxia-free-face-shield"]


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


# --- Groups + trim restructure (dark) -------------------------------------

def test_trimmed_slugs_are_gone_from_secondary_tools():
    slugs = [t["slug"] for t in orm.SECONDARY_TOOLS]
    for removed in _REMOVED_SLUGS:
        assert removed not in slugs


def test_trimmed_slugs_are_gone_from_built_roadmap():
    rm = orm.build_roadmap(owned_slugs=set())
    slugs = [x["slug"] for x in rm]
    for removed in _REMOVED_SLUGS:
        assert removed not in slugs


def test_every_secondary_item_has_a_valid_category():
    assert orm.SECONDARY_TOOLS  # sanity: not accidentally emptied
    for tool in orm.SECONDARY_TOOLS:
        assert tool.get("category") in orm.CATEGORY_ORDER


def test_every_built_secondary_item_carries_its_category():
    rm = orm.build_roadmap(owned_slugs=set())
    secondary = [x for x in rm if x["tier"] == "secondary"]
    assert secondary
    for item in secondary:
        assert item.get("category") in orm.CATEGORY_ORDER


def test_nes_mihealth_appears_exactly_twice_pemf_and_microcurrent():
    entries = [t for t in orm.SECONDARY_TOOLS if t["slug"] == "nes-mihealth"]
    assert len(entries) == 2
    categories = sorted(t["category"] for t in entries)
    assert categories == ["Microcurrent", "PEMF"]


def test_nes_mihealth_duplicate_survives_into_built_roadmap():
    rm = orm.build_roadmap(owned_slugs=set())
    entries = [x for x in rm if x["slug"] == "nes-mihealth"]
    assert len(entries) == 2
    categories = sorted(x["category"] for x in entries)
    assert categories == ["Microcurrent", "PEMF"]


def test_owning_nes_mihealth_removes_both_entries():
    rm = orm.build_roadmap(owned_slugs={"nes-mihealth"})
    slugs = [x["slug"] for x in rm]
    assert "nes-mihealth" not in slugs
    assert slugs.count("nes-mihealth") == 0


def test_unique_secondary_slug_count_is_41():
    unique_slugs = {t["slug"] for t in orm.SECONDARY_TOOLS}
    assert len(unique_slugs) == 41


def test_heroes_still_lead_after_restructure():
    rm = orm.build_roadmap(owned_slugs=set())
    slugs = [x["slug"] for x in rm]
    assert slugs[:3] == _HERO_ORDER
    hero_count = sum(1 for x in rm if x["tier"] == "hero")
    assert hero_count == 3
    # everything after the heroes is secondary
    assert all(x["tier"] == "secondary" for x in rm[3:])


def test_category_order_matches_spec():
    assert orm.CATEGORY_ORDER == [
        "Light", "Water", "Air", "PEMF", "Microcurrent", "EMF", "Sound",
        "Bioenergetic", "Detox",
    ]
