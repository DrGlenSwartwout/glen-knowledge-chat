import pytest
from dashboard.packing import BOXES_MM, fit_subset, fits_all, pack_count, split_into_boxes

# Ø×H in mm (cm×10)
BOTTLES_MM = {
    "120cap": (80, 100), "100ml": (50, 160), "30roll": (40, 100),
    "50ml": (40, 140), "30ml": (40, 110), "15ml": (30, 100), "5ml": (30, 80),
    "30g": (70, 70), "30cap": (50, 90),
}
# Verified bare-geometry counts: (S, M, L)
EXPECTED = {
    "120cap": (0, 6, 9), "100ml": (3, 10, 12), "30roll": (6, 36, 63),
    "50ml": (5, 18, 49), "30ml": (6, 36, 49), "15ml": (10, 72, 108), "5ml": (10, 84, 120),
    "30g": (0, 9, 32), "30cap": (6, 24, 36),
}

@pytest.mark.parametrize("key", list(EXPECTED))
def test_single_type_counts_match_reference(key):
    d, h = BOTTLES_MM[key]
    got = tuple(
        pack_count([(d, h)] * 500, BOXES_MM[size]) for size in ("S", "M", "L")
    )
    assert got == EXPECTED[key]


def test_mixed_load_fits_medium():
    items = [BOTTLES_MM["120cap"]] * 2 + [BOTTLES_MM["15ml"]] * 6 + [BOTTLES_MM["5ml"]] * 10
    assert fits_all(items, BOXES_MM["M"])


def test_padding_lowers_capacity():
    bare = pack_count([BOTTLES_MM["5ml"]] * 500, BOXES_MM["M"])
    padded = pack_count([BOTTLES_MM["5ml"]] * 500, BOXES_MM["M"],
                        wrap_mm=6, box_margin_mm=10)
    assert padded < bare


def test_too_wide_bottle_never_fits_small():
    # 120cap Ø80mm exceeds S's two usable cross dims at any orientation
    assert pack_count([BOTTLES_MM["120cap"]], BOXES_MM["S"]) == 0


def test_split_single_box_when_fits():
    assert split_into_boxes([BOTTLES_MM["15ml"]] * 5) == ["S"]


def test_split_picks_smallest_single_box():
    # 20 x 15ml fits L (108 cap) -> but also M (72). Smallest single box = M.
    assert split_into_boxes([BOTTLES_MM["15ml"]] * 20) == ["M"]


def test_split_into_multiple_boxes_when_oversized():
    # 200 x 15ml: L holds 108, so needs 2 boxes; last sizes down.
    boxes = split_into_boxes([BOTTLES_MM["15ml"]] * 200)
    assert boxes is not None
    assert len(boxes) == 2
    assert boxes[0] == "L"


def test_split_returns_none_when_bottle_too_big():
    # A bottle wider than every box cross-section.
    assert split_into_boxes([(200, 200)]) is None


def test_a_50ml_dropper_fits_the_small_box_at_prods_margin():
    """Glen 2026-07-09: "50 ml bottles do fit in the small box."

    They do — in PRODUCTION, where box_margin_mm is 5. The repo's default was 10, which
    seeds fresh databases (and every local sanity check) with a geometry prod does not
    use: a 35x135mm bottle plus 6mm wrap is 41x141, and Small's 50x150x230 interior less
    a 10mm margin leaves 40x140 — it misses by 1mm on each axis. Same class of bug as the
    bottle-name universes: local config that silently disagrees with prod.
    """
    from dashboard.packing import BOXES_MM, fits_all, pack_count
    from dashboard.shipping import _PACKING_DEFAULTS
    assert _PACKING_DEFAULTS["box_margin_mm"] == 5, "repo default must match prod"
    prod = {"wrap_mm": _PACKING_DEFAULTS["wrap_mm"],
            "box_margin_mm": _PACKING_DEFAULTS["box_margin_mm"]}
    assert fits_all([(35, 135)], BOXES_MM["S"], **prod), "a 50 ml dropper fits Small"
    assert pack_count([(35, 135)] * 20, BOXES_MM["S"], **prod) == 5
    # ...and a 30 ml infoceutical (40x110) still does NOT — Glen did not claim it does.
    assert not fits_all([(40, 110)], BOXES_MM["S"], **prod)
