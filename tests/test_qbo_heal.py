import datetime
import sqlite3

from dashboard import orders as O
from dashboard import qbo_heal


def _fresh_db():
    cx = sqlite3.connect(":memory:")
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    return cx


def _new_order(cx, ref, email="a@b.com"):
    return O.upsert_order(cx, source="biofield", external_ref=ref, email=email,
                          total_cents=30000)


def _mark_pending(cx, order_id, minutes_ago):
    ts = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes_ago)).isoformat()
    cx.execute("UPDATE orders SET qbo_sales_receipt_id='PENDING', updated_at=? WHERE id=?",
               (ts, order_id))
    cx.commit()


def test_case_b_stamps_existing_receipt_and_does_not_book():
    cx = _fresh_db()
    oid = _new_order(cx, "tok-b")
    _mark_pending(cx, oid, 30)

    stamp_calls = []
    book_calls = []

    def find_receipt(token, email=None, since_date=None):
        assert token == "tok-b"
        return {"Id": "SR9"}

    def stamp(cx_, order_id, receipt_id):
        stamp_calls.append((order_id, receipt_id))

    def book(cx_, order):
        book_calls.append(order)
        return "SHOULD-NOT-BOOK"

    out = qbo_heal.heal_pending_receipts(cx, find_receipt=find_receipt, book=book, stamp=stamp)

    assert stamp_calls == [(oid, "SR9")]
    assert book_calls == []
    assert out == [{"order_id": oid, "action": "stamped", "receipt_id": "SR9"}]


def test_case_a_no_receipt_clears_then_rebooks():
    cx = _fresh_db()
    oid = _new_order(cx, "tok-a")
    _mark_pending(cx, oid, 30)

    seen_at_book_time = {}
    stamp_calls = []

    def find_receipt(token, email=None, since_date=None):
        return None

    def book(cx_, order):
        # by the time book() is called, the PENDING must already be cleared
        seen_at_book_time["qbo_sales_receipt_id"] = order.get("qbo_sales_receipt_id")
        row = cx_.execute("SELECT qbo_sales_receipt_id FROM orders WHERE id=?",
                          (order["id"],)).fetchone()
        seen_at_book_time["db_value"] = row[0]
        return "SR-new"

    def stamp(cx_, order_id, receipt_id):
        stamp_calls.append((order_id, receipt_id))

    out = qbo_heal.heal_pending_receipts(cx, find_receipt=find_receipt, book=book, stamp=stamp)

    assert seen_at_book_time["qbo_sales_receipt_id"] is None
    assert seen_at_book_time["db_value"] is None
    assert stamp_calls == []  # heal sweep does not call stamp in Case A; book() does its own stamping
    assert out == [{"order_id": oid, "action": "rebooked", "receipt_id": "SR-new"}]


def test_find_receipt_raises_skips_order_leaves_pending():
    cx = _fresh_db()
    oid = _new_order(cx, "tok-raise")
    _mark_pending(cx, oid, 30)

    def find_receipt(token, email=None, since_date=None):
        raise RuntimeError("QBO transient error")

    def book(cx_, order):
        raise AssertionError("book must not be called when find_receipt raises")

    def stamp(cx_, order_id, receipt_id):
        raise AssertionError("stamp must not be called when find_receipt raises")

    out = qbo_heal.heal_pending_receipts(cx, find_receipt=find_receipt, book=book, stamp=stamp)

    assert out == []
    row = cx.execute("SELECT qbo_sales_receipt_id FROM orders WHERE id=?", (oid,)).fetchone()
    assert row[0] == "PENDING"


def test_age_guard_skips_recently_pending_order():
    cx = _fresh_db()
    oid = _new_order(cx, "tok-recent")
    _mark_pending(cx, oid, 2)  # only 2 minutes ago -- inside the 10-min guard window

    def find_receipt(token, email=None, since_date=None):
        raise AssertionError("find_receipt must not be called for a too-recent PENDING order")

    def book(cx_, order):
        raise AssertionError("book must not be called")

    def stamp(cx_, order_id, receipt_id):
        raise AssertionError("stamp must not be called")

    out = qbo_heal.heal_pending_receipts(cx, find_receipt=find_receipt, book=book, stamp=stamp)

    assert out == []
    row = cx.execute("SELECT qbo_sales_receipt_id FROM orders WHERE id=?", (oid,)).fetchone()
    assert row[0] == "PENDING"


def test_non_pending_orders_never_touched():
    cx = _fresh_db()
    oid_real = _new_order(cx, "tok-real")
    O.set_order_sales_receipt_id(cx, oid_real, "SR-already")
    cx.execute("UPDATE orders SET updated_at=? WHERE id=?",
               ((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=30)).isoformat(),
                oid_real))
    cx.commit()

    oid_null = _new_order(cx, "tok-null")
    cx.execute("UPDATE orders SET updated_at=? WHERE id=?",
               ((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=30)).isoformat(),
                oid_null))
    cx.commit()

    def find_receipt(token, email=None, since_date=None):
        raise AssertionError("find_receipt must not be called for non-PENDING orders")

    def book(cx_, order):
        raise AssertionError("book must not be called")

    def stamp(cx_, order_id, receipt_id):
        raise AssertionError("stamp must not be called")

    out = qbo_heal.heal_pending_receipts(cx, find_receipt=find_receipt, book=book, stamp=stamp)

    assert out == []
    assert O.get_order(cx, oid_real)["qbo_sales_receipt_id"] == "SR-already"
    assert O.get_order(cx, oid_null)["qbo_sales_receipt_id"] is None
