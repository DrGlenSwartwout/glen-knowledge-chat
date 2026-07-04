import sqlite3
from dashboard import orders as o


def _cx():
    cx = sqlite3.connect(":memory:")
    o.init_orders_table(cx)
    return cx


def test_margin_cents_column_exists():
    cols = {r[1] for r in _cx().execute("PRAGMA table_info(orders)")}
    assert "margin_cents" in cols


def test_upsert_writes_margin_on_insert():
    cx = _cx()
    o.upsert_order(cx, source="dispensary", external_ref="INV1", total_cents=100, margin_cents=2000)
    assert cx.execute("SELECT margin_cents FROM orders WHERE external_ref='INV1'").fetchone()[0] == 2000


def test_upsert_none_margin_does_not_clobber():
    cx = _cx()
    o.upsert_order(cx, source="dispensary", external_ref="INV2", margin_cents=2000, total_cents=100)
    o.upsert_order(cx, source="dispensary", external_ref="INV2", total_cents=200)  # margin omitted
    row = cx.execute("SELECT margin_cents, total_cents FROM orders WHERE external_ref='INV2'").fetchone()
    assert row == (2000, 200)
