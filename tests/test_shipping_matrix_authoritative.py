"""The capacity matrix is authoritative wherever Rae has filled it in — even for
a bottle type that ALSO carries dimensions.

Regression: a 30-cap FF bottle is Ø51 x H90, but the Small USPS box's smallest
interior is 50 mm, so the geometric packer places 0 in Small and boxed every
1-6 bottle order as Medium ($23 vs the $13 Small the matrix intends). The matrix
row (S=6, M=12, L=20) is the operator's ground truth and must win.

init_shipping_schema seeds the standard bottles WITH dims but no capacity rows,
so these tests add only the capacity rows (as Rae does at /admin/shipping).
"""
import sqlite3

from dashboard.shipping import (
    init_shipping_schema, set_box_capacity, pick_boxes, quote,
)


def _db(tmp_path):
    db_path = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db_path) as cx:
        init_shipping_schema(cx)
    return db_path


def _bid(db_path, name):
    with sqlite3.connect(db_path) as cx:
        return cx.execute("SELECT id FROM bottle_types WHERE name=?", (name,)).fetchone()[0]


def _fill_caps(db_path, name, s, m, l):
    bid = _bid(db_path, name)
    set_box_capacity(bid, "S", s, db_path=db_path)
    set_box_capacity(bid, "M", m, db_path=db_path)
    set_box_capacity(bid, "L", l, db_path=db_path)


def test_matrix_wins_over_geometry_for_dimmed_type(tmp_path):
    db = _db(tmp_path)
    _fill_caps(db, "30 Caps", 6, 12, 20)  # seeded with dims Ø51xH90
    # 3 bottles: matrix says Small (3/6). Geometry (Ø51 > 50mm) would say Medium.
    assert pick_boxes({"30 Caps": 3}, db_path=db) == ["S"]


def test_matrix_thresholds(tmp_path):
    db = _db(tmp_path)
    _fill_caps(db, "30 Caps", 6, 12, 20)
    assert pick_boxes({"30 Caps": 6}, db_path=db) == ["S"]    # exactly full Small
    assert pick_boxes({"30 Caps": 7}, db_path=db) == ["M"]    # spills to Medium
    assert pick_boxes({"30 Caps": 12}, db_path=db) == ["M"]   # full Medium
    assert pick_boxes({"30 Caps": 13}, db_path=db) == ["L"]   # spills to Large
    assert pick_boxes({"30 Caps": 20}, db_path=db) == ["L"]   # full Large


def test_large_order_defers_to_geometry(tmp_path):
    """A load too big for a single matrix box falls through to geometry (a dimmed
    type still gets a valid box list — the matrix override only governs the
    single-box selection, it doesn't drop the geometric multi-box path)."""
    db = _db(tmp_path)
    _fill_caps(db, "30 Caps", 6, 12, 20)
    boxes = pick_boxes({"30 Caps": 25}, db_path=db)  # 25 > Large cap 20
    assert boxes and all(b in ("S", "M", "L") for b in boxes)


def test_quote_charges_small_rate_for_small_order(tmp_path):
    db = _db(tmp_path)
    _fill_caps(db, "30 Caps", 6, 12, 20)
    q = quote({"30 Caps": 3}, db_path=db)
    assert q["box_size"] == "S"
    assert q["shipping_cents"] == 1300   # the Small rate, not Medium's 2300


def test_geometry_still_used_when_no_matrix_row(tmp_path):
    """A dims-only bottle (Rae hasn't filled its matrix row) still packs
    geometrically — the change only flips precedence, it doesn't drop geometry."""
    db = _db(tmp_path)
    # Seeded 5 mL dropper: dims Ø23xH75, NO capacity rows.
    assert pick_boxes({"Dropper 5 mL": 1}, db_path=db) == ["S"]


def test_mixed_all_capped_uses_matrix(tmp_path):
    """A cart of two types that BOTH have matrix rows uses the matrix (fractional
    fill)."""
    db = _db(tmp_path)
    _fill_caps(db, "30 Caps", 6, 12, 20)
    _fill_caps(db, "Dropper 30 mL", 8, 30, 50)
    # 3 caps (3/6) + 2 droppers (2/8) = 0.75 of Small -> one Small box.
    assert pick_boxes({"30 Caps": 3, "Dropper 30 mL": 2}, db_path=db) == ["S"]
