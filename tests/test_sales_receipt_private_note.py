import sqlite3
import pytest
from dashboard import qbo_billing as qb
from dashboard import qbo_sale
from dashboard import orders as O


@pytest.fixture
def captured(monkeypatch):
    """Capture the body posted to _post, stubbing all QBO I/O."""
    sink = {}

    def fake_post(path, body):
        sink["path"] = path
        sink["body"] = body
        return {"SalesReceipt": {"Id": "SR1", "DocNumber": "1001"}}

    monkeypatch.setattr(qb, "_post", fake_post)
    monkeypatch.setattr(qb, "_first_bank_account_id", lambda: "BANK9")
    monkeypatch.setattr(qb, "find_or_create_item", lambda name, price=None: {"Id": "IT1"})
    return sink


def test_private_note_stamped_when_given(captured):
    qb.create_sales_receipt({"Id": "C1"}, [{"name": "Widget", "amount": 10.0, "qty": 1}],
                            private_note="order:tok1")
    assert captured["body"]["PrivateNote"] == "order:tok1"


def test_private_note_omitted_when_not_given(captured):
    qb.create_sales_receipt({"Id": "C1"}, [{"name": "Widget", "amount": 10.0, "qty": 1}])
    assert "PrivateNote" not in captured["body"]


class _FakeQB:
    def __init__(self):
        self.receipts = 0
        self.last_private_note = None

    def find_or_create_customer(self, email, name=""):
        return {"Id": "C1"}

    def create_sales_receipt(self, cust, lines, *, discount_cents=0, tax_cents=0,
                             email_to=None, private_note=None):
        self.receipts += 1
        self.last_private_note = private_note
        return {"Id": "SR1"}


def _db(tmp_path):
    cx = sqlite3.connect(str(tmp_path / "orders.db"))
    cx.row_factory = sqlite3.Row
    O.init_orders_table(cx)
    return cx


def test_book_sale_on_payment_stamps_order_token_private_note(tmp_path, monkeypatch):
    cx = _db(tmp_path)
    oid = O.upsert_order(cx, source="funnel", external_ref="tok-abc", email="a@b.com",
                         total_cents=30000)
    payload = {"lines": [{"name": "X", "amount": 300.0, "qty": 1}],
               "discount_cents": 0, "tax_cents": 0}
    O.set_order_qbo_lines(cx, "tok-abc", payload)

    fake = _FakeQB()
    monkeypatch.setattr(qbo_sale, "qbo_billing", fake)

    order = O.get_order(cx, oid)
    sr = qbo_sale.book_sale_on_payment(cx, order)

    assert sr == "SR1"
    assert fake.receipts == 1
    assert fake.last_private_note == f"order:{order.get('external_ref')}"
    assert fake.last_private_note == "order:tok-abc"
