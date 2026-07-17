import sqlite3
from dashboard import orders

def _mk():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    orders.init_orders_table(cx)
    return cx

def _seed(cx):
    cx.execute("INSERT INTO orders (created_at, source, external_ref, email, status) VALUES (?,?,?,?,?)",
               (orders._now(), "funnel", "tok1", "a@b.com", "new"))
    cx.commit()
    return cx.execute("SELECT id FROM orders WHERE external_ref='tok1'").fetchone()["id"]

def test_settled_at_column_exists_and_defaults_null():
    cx = _mk(); oid = _seed(cx)
    row = cx.execute("SELECT settled_at FROM orders WHERE id=?", (oid,)).fetchone()
    assert row["settled_at"] is None

def test_mark_order_settled_sets_once_and_is_idempotent():
    cx = _mk(); oid = _seed(cx)
    assert orders.mark_order_settled(cx, oid) is True
    row = cx.execute("SELECT settled_at FROM orders WHERE id=?", (oid,)).fetchone()
    assert row["settled_at"] is not None
    first = row["settled_at"]
    # second call is a no-op: does not overwrite, returns False
    assert orders.mark_order_settled(cx, oid) is False
    row2 = cx.execute("SELECT settled_at FROM orders WHERE id=?", (oid,)).fetchone()
    assert row2["settled_at"] == first

def test_find_order_exposes_settled_at():
    cx = _mk(); _seed(cx)
    o = orders.find_order_by_external_ref(cx, "tok1")
    assert "settled_at" in dict(o)
