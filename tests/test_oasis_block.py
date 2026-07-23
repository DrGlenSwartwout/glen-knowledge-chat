import sqlite3

from dashboard import oasis_block, orders as _o, owned_tools as ot, wishlist as wl


def _cx():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    ot.init_table(cx)
    _o.init_orders_table(cx)
    return cx


def test_block_off_when_disabled():
    assert oasis_block.build_block(None, "a@b.com", False) == {"enabled": False}


def test_owned_external_tool_excluded_from_roadmap():
    cx = _cx()
    ot.add(cx, "a@b.com", "Water Ionizer", "OtherCo", slug="water-ionizer")
    blk = oasis_block.build_block(cx, "a@b.com", True, terrain_phase="cleanse")
    road_slugs = {r["slug"] for r in blk["build_out"]["roadmap"]}
    assert "water-ionizer" not in road_slugs                       # owned externally -> not recommended
    assert {"name", "brand"} <= set(blk["build_out"]["owned_external"][0].keys())


# --- Mismatch 1: real catalog device-slug families vs. the roadmap's
# simplified hero slugs. Owning ANY real variant must exclude the hero. ---

def test_owned_external_real_variant_slug_excludes_hero():
    """Client's self-reported tool uses the REAL catalog variant slug
    (water-ionizer-5plate), not the roadmap's simplified "water-ionizer"."""
    cx = _cx()
    ot.add(cx, "a@b.com", "5-Plate Water Ionizer", "Living Water",
           slug="water-ionizer-5plate")
    blk = oasis_block.build_block(cx, "a@b.com", True)
    road_slugs = {r["slug"] for r in blk["build_out"]["roadmap"]}
    assert "water-ionizer" not in road_slugs


def test_ordered_real_variant_slug_excludes_hero():
    """Client ORDERED the real catalog variant (water-ionizer-9plate) -- it
    must surface in owned_from_us AND exclude the "water-ionizer" hero."""
    cx = _cx()
    _o.upsert_order(cx, source="test", external_ref="ord-1", email="a@b.com",
                    items=[{"slug": "water-ionizer-9plate", "qty": 1}],
                    total_cents=100000)
    blk = oasis_block.build_block(cx, "a@b.com", True)
    road_slugs = {r["slug"] for r in blk["build_out"]["roadmap"]}
    assert "water-ionizer" not in road_slugs
    us_slugs = {d["slug"] for d in blk["build_out"]["owned_from_us"]}
    assert "water-ionizer-9plate" in us_slugs


def test_normalize_owned_for_roadmap_family_map():
    norm = oasis_block._normalize_owned_for_roadmap(
        {"water-ionizer-9plate", "harmony-laser", "kloud-pemf-mini", "some-other-slug"})
    assert norm == {"water-ionizer", "harmony", "kloud", "some-other-slug"}


def test_harmony_consumable_slug_does_not_map_to_hero():
    # A non-device "harmony-*" consumable (self-reported via owned_tools) must NOT
    # be mapped onto the "harmony" hero -- only "harmony-laser*" is the device.
    norm = oasis_block._normalize_owned_for_roadmap(
        {"harmony-flower-essence-in-terrain-restore"})
    assert "harmony" not in norm
    assert norm == {"harmony-flower-essence-in-terrain-restore"}


# --- Task 7: build_out.wanted -- the shared wishlist reflection ---

def test_build_out_wanted_resolves_wishlisted_slug_to_name_and_url():
    """A slug added to the client's wishlist (whether via this tile's own
    "Add to wishlist" roadmap action or My Remedies' "Add to my Oasis") must
    surface, resolved to {slug, name, url}, in build_out.wanted."""
    cx = _cx()
    wl.init_wishlist_table(cx)
    wl.toggle(cx, wl.resolve_owner("a@b.com", None), "aces-eyedrops")
    blk = oasis_block.build_block(cx, "a@b.com", True)
    wanted = blk["build_out"]["wanted"]
    assert len(wanted) == 1
    assert wanted[0]["slug"] == "aces-eyedrops"
    assert wanted[0]["name"] == "ACES Eyedrops"
    assert wanted[0]["url"] == "/begin/product/aces-eyedrops"


def test_build_out_wanted_skips_slug_not_in_catalog_or_roadmap():
    """A wishlisted slug that resolves in NEITHER the purchasable catalog NOR
    any roadmap table (HERO_TOOLS / TERRAIN_TOOLS / GENERAL_TOOLS) -- i.e. a
    removed/renamed product or a typo -- is silently skipped rather than
    shown with placeholder text."""
    cx = _cx()
    wl.init_wishlist_table(cx)
    wl.toggle(cx, wl.resolve_owner("a@b.com", None), "not-a-real-slug")
    blk = oasis_block.build_block(cx, "a@b.com", True)
    assert blk["build_out"]["wanted"] == []


def test_build_out_wanted_empty_when_no_wishlist():
    blk = oasis_block.build_block(_cx(), "a@b.com", True)
    assert blk["build_out"]["wanted"] == []


# --- Final-review Fix 1: owned_from_us is a DEVICE allowlist, not "not a
# consumable" (that complement also includes services/consults, info_only,
# digital ebooks, and print books). ---

def test_owned_from_us_only_devices_not_services_or_books_or_consumables():
    cx = _cx()
    _o.upsert_order(
        cx, source="test", external_ref="ord-devices", email="a@b.com",
        items=[
            {"slug": "harmony-laser", "qty": 1},       # real device (bottle_type harmony-laser)
            {"slug": "biofield-analysis", "qty": 1},   # service/consult (info_only+service)
            {"slug": "book-refreshing-vision", "qty": 1},  # print book (bottle_type book)
            {"slug": "aces-eyedrops", "qty": 1},       # dosed consumable (Dropper 5 mL)
        ],
        total_cents=100000,
    )
    blk = oasis_block.build_block(cx, "a@b.com", True)

    us_slugs = {d["slug"] for d in blk["build_out"]["owned_from_us"]}
    assert us_slugs == {"harmony-laser"}

    replenish_slugs = {r["slug"] for r in blk["replenish"]}
    assert replenish_slugs == {"aces-eyedrops"}


# --- Final-review Fix 2: wishlisting an off-catalog roadmap slug (hero/
# terrain/general) resolves via the roadmap tables instead of no-oping. ---

def test_wanted_resolves_roadmap_hero_slug_to_its_real_name():
    from dashboard import oasis_roadmap

    cx = _cx()
    wl.init_wishlist_table(cx)
    wl.toggle(cx, wl.resolve_owner("a@b.com", None), "harmony")
    blk = oasis_block.build_block(cx, "a@b.com", True)

    wanted = blk["build_out"]["wanted"]
    assert len(wanted) == 1
    hero_name = next(t["name"] for t in oasis_roadmap.HERO_TOOLS if t["slug"] == "harmony")
    assert wanted[0] == {"slug": "harmony", "name": hero_name, "url": "/begin/product/harmony"}
