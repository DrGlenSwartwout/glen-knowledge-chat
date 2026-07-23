import sqlite3
from dashboard import order_payments as op

def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    op.ensure_table(cx)
    cx.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, email TEXT, name TEXT, channel TEXT)")
    cx.execute("INSERT INTO orders (id, email, name, channel) VALUES (1,'michael@x.com','Michael','web')")
    return cx

def test_add_payment_without_payer_is_null():
    cx = _cx()
    row = op.add_payment(cx, 1, 5000, "Zelle")
    assert row["payer_email"] is None

def test_add_payment_stamps_payer():
    cx = _cx()
    row = op.add_payment(cx, 1, 5000, "Zelle", payer_email="steve@x.com")
    assert row["payer_email"] == "steve@x.com"
