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

def test_money_view_attributes_to_payer():
    cx = _cx()
    op.add_payment(cx, 1, 5000, "Zelle", payer_email="steve@x.com")
    op.add_payment(cx, 1, 2000, "Zelle")  # self-paid, payer NULL
    cx.row_factory = sqlite3.Row
    rows = op.ledger_rows_for_payments_view(cx)
    emails = sorted(r["email"] for r in rows)
    assert emails == ["michael@x.com", "steve@x.com"]

def test_refund_inherits_payer():
    cx = _cx()
    pay = op.add_payment(cx, 1, 5000, "Zelle", payer_email="steve@x.com")
    op.add_refund(cx, 1, 5000, "Zelle", refunds_payment_id=pay["id"])
    ref = cx.execute("SELECT payer_email FROM order_payments WHERE kind='refund'").fetchone()
    assert ref[0] == "steve@x.com"

def test_caregiver_payers_for_lists_foreign_payers():
    cx = _cx()
    op.add_payment(cx, 1, 3000, "Zelle", payer_email="steve@x.com")
    op.add_payment(cx, 1, 2000, "Zelle")  # self-paid → not a caregiver payer
    assert op.caregiver_payers_for(cx, 1, "michael@x.com") == ["steve@x.com"]
