import sqlite3
import tempfile
import os
from dashboard import practitioner_pricing as pp

def _cx():
    return sqlite3.connect(":memory:")


def test_get_config_defaults_empty():
    with _cx() as cx:
        assert pp.get_config(cx, "7") == {}


def test_set_then_get_roundtrips():
    cfg = {"standard": {"same_sku": {"enabled": True, "dial": 0.5}}}
    with _cx() as cx:
        pp.set_config(cx, "7", cfg)
        assert pp.get_config(cx, "7") == cfg


def test_set_config_upserts():
    with _cx() as cx:
        pp.set_config(cx, "7", {"standard": {"same_sku": {"enabled": True, "dial": 0.2}}})
        pp.set_config(cx, "7", {"standard": {"same_sku": {"enabled": False, "dial": 0.9}}})
        assert pp.get_config(cx, "7")["standard"]["same_sku"]["dial"] == 0.9


def test_defaults():
    s = pp.load_settings({})
    assert s["fee_pct"] == 0.33
    assert s["map_default_cents"] == 6700

def test_drop_ship_base_matches_blended_curve():
    # q1 = $50 for everyone; 12 uncertified ~ $47.11; 12 fully-certified ~ $42.76
    assert pp.drop_ship_base_cents(1, 0) == 5000
    assert pp.drop_ship_base_cents(1, 12) == 5000
    assert pp.drop_ship_base_cents(12, 0) == 4711
    assert pp.drop_ship_base_cents(12, 12) == 4276
    assert pp.drop_ship_base_cents(40, 12) == 2500   # floor at 2B, fully certified

def test_service_fee_flat_33pct_of_markup():
    s = pp.load_settings({})
    assert pp.service_fee_cents(8000, 5000, s) == 990    # 33% of (80-50)=30 -> $9.90
    assert pp.service_fee_cents(5000, 5000, s) == 0      # no markup -> no fee
    assert pp.service_fee_cents(4000, 5000, s) == 0      # negative markup clamps to 0

def test_resolve_selling_from_dollars_and_percent():
    # retail $70, MAP $67
    assert pp.resolve_selling_cents({"price_cents": 8000}, retail_cents=7000, map_cents=6700) == 8000
    # +20% over retail -> $84
    assert pp.resolve_selling_cents({"markup_pct": 20}, retail_cents=7000, map_cents=6700) == 8400
    # default to retail when nothing set
    assert pp.resolve_selling_cents({}, retail_cents=7000, map_cents=6700) == 7000

def test_resolve_selling_rejects_below_map():
    import pytest
    with pytest.raises(pp.MapViolation):
        pp.resolve_selling_cents({"price_cents": 6000}, retail_cents=7000, map_cents=6700)   # $60 < MAP
    with pytest.raises(pp.MapViolation):
        pp.resolve_selling_cents({"markup_pct": -10}, retail_cents=7000, map_cents=6700)      # $63 < MAP

def test_resolve_selling_edge_cases():
    # exactly at MAP is allowed (strict-less-than floor)
    assert pp.resolve_selling_cents({"price_cents": 6700}, retail_cents=7000, map_cents=6700) == 6700
    # markup_pct 0 resolves to retail (not the empty-default branch), and clears MAP here
    assert pp.resolve_selling_cents({"markup_pct": 0}, retail_cents=7000, map_cents=6700) == 7000
    # an explicit price_cents 0 is honored (not treated as unset) -> below MAP -> raises
    import pytest
    with pytest.raises(pp.MapViolation):
        pp.resolve_selling_cents({"price_cents": 0}, retail_cents=7000, map_cents=6700)

def test_companion_figure_helper():
    # given a dollar price, report the implied markup %, and vice versa, for the UI
    assert pp.markup_pct_for(8400, 7000) == 20.0
    assert pp.price_for_markup(20, 7000) == 8400

def test_margin_and_dropship_wholesale():
    s = pp.load_settings({})
    # S=$80, base=$50 -> fee $9.90 -> margin $20.10; practitioner-paid pays base+fee=$59.90
    q = pp.quote_line(selling_cents=8000, qty=1, modules=0, settings=s)
    assert q["base_cents"] == 5000
    assert q["fee_cents"] == 990
    assert q["margin_cents"] == 2010
    assert q["dropship_wholesale_cents"] == 5990       # what practitioner-paid mode pays us
    assert q["line_selling_cents"] == 8000

def test_quote_line_qty_uses_blended_volume():
    s = pp.load_settings({})
    # 12 bottles uncertified -> base $47.11/bottle; selling $80 each
    q = pp.quote_line(selling_cents=8000, qty=12, modules=0, settings=s)
    assert q["base_cents"] == 4711
    assert q["fee_cents"] == 1085          # round(0.33 * 3289) — hard value, not the formula
    assert q["margin_cents"] == 2204       # 8000 - 4711 - 1085

def test_margin_never_negative():
    s = pp.load_settings({})
    q = pp.quote_line(selling_cents=5000, qty=1, modules=0, settings=s)   # S == base, fee 0
    assert q["margin_cents"] == 0


def test_set_config_persists_without_with_block(tmp_path):
    """Regression test: set_config must commit internally.

    The real app.py callers open a connection with cx = sqlite3.connect(path)
    (without a `with` block) and close it manually. If set_config does not
    cx.commit() internally, the write is silently rolled back on close.

    This test reproduces that call pattern and asserts the config persists.
    """
    db_path = str(tmp_path / "test.db")
    cfg = {"standard": {"same_sku": {"enabled": True, "dial": 0.75}}}

    # Simulate real app.py usage: open without `with`, call set_config, close manually
    cx1 = sqlite3.connect(db_path)
    pp.set_config(cx1, "test-pid", cfg)
    cx1.close()

    # Reopen to verify persistence
    cx2 = sqlite3.connect(db_path)
    retrieved = pp.get_config(cx2, "test-pid")
    cx2.close()

    # Must be equal to what we set (not an empty dict)
    assert retrieved == cfg
