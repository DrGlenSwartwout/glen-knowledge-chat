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


def test_case_a_concurrent_winner_between_select_and_clear_is_not_double_booked():
    # Simulates a race: our sweep's SELECT read the row as PENDING and our
    # find_receipt() call returned None (Case A -- "no receipt exists yet").
    # But between that read and our clear, a concurrent/faster instance
    # resolved the SAME order and stamped a real receipt id onto it. Our
    # blind clear+book must not stomp that -- it must detect the row no
    # longer says PENDING and skip rebooking entirely.
    cx = _fresh_db()
    oid = _new_order(cx, "tok-race")
    _mark_pending(cx, oid, 30)

    book_calls = []

    def find_receipt(token, email=None, since_date=None):
        # Side effect: simulate a concurrent instance winning the race and
        # stamping a real receipt id onto this order right now.
        cx.execute("UPDATE orders SET qbo_sales_receipt_id=? WHERE id=?",
                   ("SR-concurrent-winner", oid))
        cx.commit()
        return None

    def book(cx_, order):
        book_calls.append(order)
        return "SHOULD-NOT-BOOK"

    def stamp(cx_, order_id, receipt_id):
        raise AssertionError("stamp must not be called in this scenario")

    out = qbo_heal.heal_pending_receipts(cx, find_receipt=find_receipt, book=book, stamp=stamp)

    assert book_calls == []
    row = cx.execute("SELECT qbo_sales_receipt_id FROM orders WHERE id=?", (oid,)).fetchone()
    assert row[0] == "SR-concurrent-winner"
    assert out == []


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


def test_pending_order_with_empty_string_external_ref_is_skipped():
    # orders.external_ref is NOT NULL in the schema, so the realistic token-less
    # case is an empty string (o.get("external_ref") is still falsy). The sweep
    # must skip such an order entirely -- never call find_receipt (which would
    # otherwise look up "order:None"/"order:" downstream), stamp, or book.
    #
    # NOTE: calls are recorded (not raised) on purpose. heal_pending_receipts
    # wraps the whole per-order body in a broad try/except, so a raised
    # AssertionError from inside find_receipt would be silently swallowed as
    # "inconclusive lookup" and the test would pass for the wrong reason.
    cx = _fresh_db()
    oid = _new_order(cx, "tok-will-be-cleared-2")
    _mark_pending(cx, oid, 30)
    cx.execute("UPDATE orders SET external_ref='' WHERE id=?", (oid,))
    cx.commit()

    find_receipt_calls = []
    book_calls = []
    stamp_calls = []

    def find_receipt(token, email=None, since_date=None):
        find_receipt_calls.append(token)
        return None

    def book(cx_, order):
        book_calls.append(order)
        return "SHOULD-NOT-BOOK"

    def stamp(cx_, order_id, receipt_id):
        stamp_calls.append((order_id, receipt_id))

    out = qbo_heal.heal_pending_receipts(cx, find_receipt=find_receipt, book=book, stamp=stamp)

    assert find_receipt_calls == []
    assert book_calls == []
    assert stamp_calls == []
    assert out == []
    row = cx.execute("SELECT qbo_sales_receipt_id FROM orders WHERE id=?", (oid,)).fetchone()
    assert row[0] == "PENDING"
