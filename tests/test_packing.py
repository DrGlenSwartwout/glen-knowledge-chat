import pytest
from dashboard.packing import BOXES_MM, fit_subset, fits_all, pack_count

# Ø×H in mm (cm×10)
BOTTLES_MM = {
    "120cap": (80, 100), "100ml": (50, 160), "30roll": (40, 100),
    "50ml": (40, 140), "15ml": (30, 100), "5ml": (30, 80),
    "100cos": (70, 70), "30cap": (50, 90),
}
# Verified bare-geometry counts: (S, M, L)
EXPECTED = {
    "120cap": (0, 6, 9), "100ml": (3, 10, 12), "30roll": (6, 36, 63),
    "50ml": (5, 18, 49), "15ml": (10, 72, 108), "5ml": (10, 84, 120),
    "100cos": (0, 9, 32), "30cap": (6, 24, 36),
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
