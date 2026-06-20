import sqlite3
from dashboard import sales_votes as sv
from dashboard import sales_images as si

def _cx(): return sqlite3.connect(":memory:")

def _row(cx, slug, kind, session_id):
    return cx.execute("SELECT chosen_variant, prompt_variant_id, model_id FROM sales_page_votes "
                      "WHERE product_slug=? AND kind=? AND session_id=?", (slug, kind, session_id)).fetchone()

def test_record_pick_persists_tags():
    cx = _cx()
    sv.record_pick(cx, "p", "botanical", 2, "s1", prompt_variant_id=7, model_id="imagen-4")
    assert _row(cx, "p", "botanical", "s1") == (2, 7, "imagen-4")

def test_record_pick_revote_updates_variant_and_tags():
    cx = _cx()
    sv.record_pick(cx, "p", "botanical", 1, "s1", prompt_variant_id=3, model_id="flux-1.1-pro")
    sv.record_pick(cx, "p", "botanical", 4, "s1", prompt_variant_id=9, model_id="recraft-v3")
    assert _row(cx, "p", "botanical", "s1") == (4, 9, "recraft-v3")
    # still one row for this (session, product, kind)
    n = cx.execute("SELECT COUNT(*) FROM sales_page_votes WHERE product_slug='p' AND kind='botanical' AND session_id='s1'").fetchone()[0]
    assert n == 1

def test_record_pick_backward_compatible_without_tags():
    cx = _cx()
    sv.record_pick(cx, "p", "mechanism", 1, "s1")   # 6-arg legacy call (Phase-4 style)
    assert _row(cx, "p", "mechanism", "s1") == (1, None, None)

def test_tags_for_returns_image_tags():
    cx = _cx()
    si.record_image(cx, "p", "botanical", 3, "botanical-3.png", prompt_variant_id=5, model_id="recraft-v3")
    assert si.tags_for(cx, "p", "botanical", 3) == (5, "recraft-v3")

def test_tags_for_missing_slot_returns_none_pair():
    cx = _cx()
    assert si.tags_for(cx, "p", "botanical", 2) == (None, None)
