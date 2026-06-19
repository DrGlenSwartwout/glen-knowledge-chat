import sqlite3
from dashboard import sales_image_pairs as sp
from dashboard import sales_images as si

def _cx(): return sqlite3.connect(":memory:")

def test_ensure_pair_inits_from_two_lowest_variants():
    cx = _cx()
    si.record_image(cx, "x", "botanical", 1, "botanical-1.png")
    si.record_image(cx, "x", "botanical", 2, "botanical-2.png")
    pair = sp.ensure_pair(cx, "x", "botanical", [1, 2])
    assert pair["champion_variant"] == 1 and pair["challenger_variant"] == 2
    assert pair["defenses"] == 0 and pair["converged"] is False

def test_set_and_get_pair_roundtrip():
    cx = _cx()
    sp.set_pair(cx, "x", "mechanism", champion=1, challenger=3, defenses=2, converged=True, last_render_at="T")
    g = sp.get_pair(cx, "x", "mechanism")
    assert g["champion_variant"] == 1 and g["challenger_variant"] == 3
    assert g["defenses"] == 2 and g["converged"] is True and g["last_render_at"] == "T"

def test_ensure_pair_none_when_under_two_variants():
    assert sp.ensure_pair(_cx(), "x", "botanical", [1]) is None

def test_next_variant_and_list_slugs():
    cx = _cx()
    si.record_image(cx, "x", "botanical", 1, "botanical-1.png")
    si.record_image(cx, "x", "botanical", 2, "botanical-2.png")
    assert si.next_variant(cx, "x", "botanical") == 3
    assert si.next_variant(cx, "x", "mechanism") == 1   # none yet for this kind
    assert "x" in si.list_image_slugs(cx)
