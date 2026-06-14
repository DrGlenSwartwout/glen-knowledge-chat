# tests/test_orders_pricing_columns.py
import sqlite3
from dashboard import orders as o

def _cx():
    cx = sqlite3.connect(":memory:")
    o.init_orders_table(cx)
    return cx

def test_upsert_records_discount_points_shipping():
    cx = _cx()
    o.upsert_order(cx, source="reorder", external_ref="INV1", email="a@x.com",
                   name="A", items=[{"name":"X","qty":1,"desc":"X"}], total_cents=5000,
                   address={"state":"CA"}, channel="retail", get_cents=0,
                   discount_cents=1500, points_redeemed_cents=300, shipping_cents=1265)
    row = cx.execute("SELECT discount_cents, points_redeemed_cents, shipping_cents "
                     "FROM orders WHERE external_ref='INV1'").fetchone()
    assert row == (1500, 300, 1265)

def test_upsert_defaults_new_columns_to_zero():
    cx = _cx()
    o.upsert_order(cx, source="reorder", external_ref="INV2", email="a@x.com",
                   items=[{"name":"X","qty":1,"desc":"X"}], total_cents=5000,
                   address={}, channel="retail", get_cents=0)
    row = cx.execute("SELECT discount_cents, points_redeemed_cents, shipping_cents "
                     "FROM orders WHERE external_ref='INV2'").fetchone()
    assert row == (0, 0, 0)
