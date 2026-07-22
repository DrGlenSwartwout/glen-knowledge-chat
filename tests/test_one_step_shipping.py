"""One Step tub shipping.

One Step (slug `one-step`) is a ~Ø140 x H190 mm meal-replacement tub. A true Ø140
exceeds every flat-rate box interior and rates as None (unshippable), which is why the
product had no bottle_type and could not be auto-rated. It now maps to a dedicated
`one-step` bottle type (shipping proxy Ø120 x H190) that resolves a single unit to USPS
Medium and bulk loads to Large — per Glen, who confirmed it ships Medium/Large with no
packing wrap.
"""
import json
import sqlite3
from pathlib import Path

from dashboard.shipping import init_shipping_schema, quote, resolve_bottle_type


def _db(tmp_path):
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
    return db


def test_single_one_step_rates_medium(tmp_path):
    q = quote({"one-step": 1}, db_path=_db(tmp_path))
    assert q["box_sizes"] == ["M"], q
    assert isinstance(q["shipping_cents"], int) and q["shipping_cents"] > 0


def test_two_one_step_needs_a_large(tmp_path):
    # The whole point of the M=1/L=2 cap: two tubs won't tile into one Medium.
    q = quote({"one-step": 2}, db_path=_db(tmp_path))
    assert q["box_sizes"] == ["L"], q
    assert isinstance(q["shipping_cents"], int) and q["shipping_cents"] > 0


def test_three_one_step_split_large_plus_medium(tmp_path):
    # Cap L=2 forces the split: two in a Large, the third in a Medium.
    q = quote({"one-step": 3}, db_path=_db(tmp_path))
    assert sorted(q["box_sizes"]) == ["L", "M"], q
    assert q["shipping_cents"] is not None


def test_real_diameter_breaks_bulk_geometry(tmp_path):
    # The proxy Ø120 still earns its keep for 3+ (the multi-box geometric split): at the
    # true Ø140 the tub fits no box, so a 3-unit order can't be packed at all.
    db = _db(tmp_path)
    with sqlite3.connect(db) as cx:
        cx.execute("UPDATE bottle_types SET diameter_mm=140 WHERE name='one-step'")
        cx.commit()
    assert quote({"one-step": 3}, db_path=db)["shipping_cents"] is None


def test_product_catalog_maps_one_step_bottle_type():
    # The shipped catalog must carry the bottle_type so _price_cart resolves it.
    data = json.loads((Path(__file__).resolve().parent.parent / "data" / "products.json").read_text())
    assert data["products"]["one-step"].get("bottle_type") == "one-step"
    # And the resolver honors that product field (no per-slug override needed).
    assert resolve_bottle_type("one-step", {"slug": "one-step", "bottle_type": "one-step"}) == "one-step"


def test_backfills_onto_existing_catalog(tmp_path):
    # Simulate a legacy (non-empty) prod catalog missing the one-step row, then re-init:
    # the targeted backfill must add it (mirrors the 30ml ensure-insert).
    db = str(tmp_path / "chat_log.db")
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
        cx.execute("DELETE FROM bottle_types WHERE name='one-step'")  # cascades caps
        cx.commit()
    with sqlite3.connect(db) as cx:
        init_shipping_schema(cx)
        row = cx.execute(
            "SELECT diameter_mm, height_mm FROM bottle_types WHERE name='one-step'"
        ).fetchone()
        caps = dict(cx.execute(
            "SELECT bc.box_size, bc.qty FROM box_capacity bc "
            "JOIN bottle_types bt ON bt.id=bc.bottle_type_id WHERE bt.name='one-step'"
        ).fetchall())
    assert row == (120, 190)
    assert caps == {"M": 1, "L": 2}          # caps backfilled alongside the dims
