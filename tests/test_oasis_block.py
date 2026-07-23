import sqlite3

from dashboard import oasis_block, orders as _o, owned_tools as ot


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
