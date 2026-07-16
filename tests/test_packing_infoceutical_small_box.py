"""Ø40 glass droppers (30ml infoceutical, 30roll) must pack into the Small box.
At the old wrap_mm=6 they came to 46mm and missed Small's 45mm usable width by 1mm,
bumping to Medium (inflated shipping). Glen confirmed they physically fit Small;
wrap_mm=5 resolves them. Bottle dims are unchanged (glass — real measured sizes)."""
from dashboard import packing, shipping

BOX_S = packing.BOXES_MM["S"]


def _fits_small(d, h, wrap):
    return packing.fits_all([(d, h)], BOX_S, wrap_mm=wrap, box_margin_mm=5)


def test_default_wrap_is_5():
    assert shipping._PACKING_DEFAULTS["wrap_mm"] == 5


def test_infoceutical_droppers_fit_small_at_wrap_5():
    assert _fits_small(40, 110, 5) is True   # 30ml infoceutical dropper
    assert _fits_small(40, 100, 5) is True   # 30 ml roll-on (30roll)


def test_droppers_missed_small_at_old_wrap_6():
    # Regression guard: documents WHY wrap dropped to 5 (a 1mm boundary miss).
    assert _fits_small(40, 110, 6) is False
    assert _fits_small(40, 100, 6) is False


def test_larger_bottles_do_not_wrongly_drop_to_small_at_wrap_5():
    assert _fits_small(72, 100, 5) is False   # 120 caps
    assert _fits_small(93, 200, 5) is False   # 360 caps
    assert _fits_small(65, 75, 5) is False    # 30 g jar
