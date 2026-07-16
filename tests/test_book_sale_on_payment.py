import json, sqlite3, types
import pytest
from dashboard import qbo_sale
from dashboard import orders as O


class _FakeQB:
    def __init__(self):
        self.receipts = 0
        self.last_lines = None
    def find_or_create_customer(self, email, name=""):
        return {"Id": "C1"}
    def create_sales_receipt(self, cust, lines, *, discount_cents=0, tax_cents=0, email_to=None):
        self.receipts += 1
        self.last_lines = lines
        self.last_discount = discount_cents
        self.last_tax = tax_cents
        return {"Id": "SR1"}


def _db(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "orders.db"))
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    return cx


def _seed(cx, ref="tok1", **lines_kw):
    oid = O.upsert_order(cx, source="funnel", external_ref=ref, email="a@b.com",
                         total_cents=30000)
    payload = {"lines": [{"name": "X", "amount": 300.0, "qty": 1}],
               "discount_cents": 1500, "tax_cents": 0}
    payload.update(lines_kw)
    O.set_order_qbo_lines(cx, ref, payload)
    return oid


def test_books_line_faithful_receipt_and_stamps(tmp_path, monkeypatch):
    cx = _db(tmp_path)
    oid = _seed(cx)
    qb = _FakeQB()
    monkeypatch.setattr(qbo_sale, "qbo_billing", qb)

    order = O.get_order(cx, oid)
    sr = qbo_sale.book_sale_on_payment(cx, order)

    assert sr == "SR1"
    assert qb.receipts == 1
    assert qb.last_lines == [{"name": "X", "amount": 300.0, "qty": 1}]
    assert qb.last_discount == 1500
    assert O.get_order(cx, oid)["qbo_sales_receipt_id"] == "SR1"


def test_double_book_prevented_on_refresh(tmp_path, monkeypatch):
    """The point of this task: two callers each holding their OWN freshly re-read
    (and still stale/unbooked) order dict -- as would happen with a checkout-return
    page refresh, a webhook+redirect race, or an alt-pay+card race that both read
    the order before either finished booking -- must only ever produce ONE Sales
    Receipt. The decision has to come from the DB row via the atomic claim, not
    from either caller's (possibly stale) dict."""
    cx = _db(tmp_path)
    oid = _seed(cx, external_ref="tok2")
    qb = _FakeQB()
    monkeypatch.setattr(qbo_sale, "qbo_billing", qb)

    # Both callers read the order BEFORE either one has booked (the race window).
    order1 = O.get_order(cx, oid)
    order2 = O.get_order(cx, oid)

    sr1 = qbo_sale.book_sale_on_payment(cx, order1)
    sr2 = qbo_sale.book_sale_on_payment(cx, order2)  # loses the claim -> must not book

    assert sr1 == "SR1"
    assert sr2 is None  # second caller lost the claim -> must not book again
    assert qb.receipts == 1
    assert O.get_order(cx, oid)["qbo_sales_receipt_id"] == "SR1"


def test_double_book_prevented_on_sequential_refresh(tmp_path, monkeypatch):
    """A plain checkout-return page refresh: the second call re-reads the order
    AFTER the first has fully booked, so it sees the real stamped id and correctly
    short-circuits via the existing-id check. Still only one receipt."""
    cx = _db(tmp_path)
    oid = _seed(cx, external_ref="tok2b")
    qb = _FakeQB()
    monkeypatch.setattr(qbo_sale, "qbo_billing", qb)

    order1 = O.get_order(cx, oid)
    sr1 = qbo_sale.book_sale_on_payment(cx, order1)

    order2 = O.get_order(cx, oid)  # fresh re-read after the first call completed
    sr2 = qbo_sale.book_sale_on_payment(cx, order2)

    assert sr1 == "SR1"
    assert sr2 == "SR1"  # already booked -> returns the existing id, no new receipt
    assert qb.receipts == 1
    assert O.get_order(cx, oid)["qbo_sales_receipt_id"] == "SR1"


def test_idempotent_no_rebook_when_already_booked(monkeypatch):
    qb = _FakeQB()
    monkeypatch.setattr(qbo_sale, "qbo_billing", qb)
    order = {"id": 5, "email": "a@b.com", "name": "A",
             "qbo_sales_receipt_id": "SRX", "qbo_lines_json": None}
    out = qbo_sale.book_sale_on_payment(None, order)
    assert out == "SRX"
    assert qb.receipts == 0


def test_pending_claim_in_flight_never_books(monkeypatch):
    qb = _FakeQB()
    monkeypatch.setattr(qbo_sale, "qbo_billing", qb)
    order = {"id": 5, "email": "a@b.com", "name": "A",
             "qbo_sales_receipt_id": "PENDING", "qbo_lines_json": None}
    out = qbo_sale.book_sale_on_payment(None, order)
    assert out is None
    assert qb.receipts == 0


def test_best_effort_never_raises(tmp_path, monkeypatch):
    cx = _db(tmp_path)
    oid = _seed(cx, external_ref="tok3")

    def boom(*a, **k):
        raise RuntimeError("QBO down")
    monkeypatch.setattr(qbo_sale, "qbo_billing",
                        types.SimpleNamespace(find_or_create_customer=boom,
                                              create_sales_receipt=boom))
    order = O.get_order(cx, oid)
    assert qbo_sale.book_sale_on_payment(cx, order) is None


def test_missing_lines_json_is_a_noop(tmp_path, monkeypatch):
    cx = _db(tmp_path)
    oid = O.upsert_order(cx, source="funnel", external_ref="tok4", email="a@b.com",
                         total_cents=30000)
    qb = _FakeQB()
    monkeypatch.setattr(qbo_sale, "qbo_billing", qb)

    order = O.get_order(cx, oid)  # no qbo_lines_json ever set
    assert qbo_sale.book_sale_on_payment(cx, order) is None
    assert qb.receipts == 0
    # the slot must NOT have been claimed since there was nothing to book
    assert O.get_order(cx, oid)["qbo_sales_receipt_id"] is None
