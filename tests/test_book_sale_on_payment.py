import json, sqlite3, types
import pytest
from dashboard import qbo_sale


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
        return {"Id": f"SR{self.receipts}"}


def _order(**kw):
    base = {"id": 5, "email": "a@b.com", "name": "A", "qbo_sales_receipt_id": None,
            "qbo_lines_json": json.dumps({"lines": [{"name": "Biofield", "amount": 300.0, "qty": 1}],
                                          "discount_cents": 1500, "tax_cents": 0})}
    base.update(kw); return base


def test_books_line_faithful_receipt_and_stamps(monkeypatch):
    qb = _FakeQB()
    stamped = {}
    monkeypatch.setattr(qbo_sale, "qbo_billing", qb)
    monkeypatch.setattr(qbo_sale, "orders",
                        types.SimpleNamespace(set_order_sales_receipt_id=
                        lambda cx, oid, sr: stamped.setdefault(oid, sr)))
    sr = qbo_sale.book_sale_on_payment(None, _order())
    assert sr == "SR1"
    assert qb.receipts == 1
    assert qb.last_lines == [{"name": "Biofield", "amount": 300.0, "qty": 1}]
    assert qb.last_discount == 1500
    assert stamped == {5: "SR1"}


def test_idempotent_no_rebook_when_already_booked(monkeypatch):
    qb = _FakeQB()
    monkeypatch.setattr(qbo_sale, "qbo_billing", qb)
    monkeypatch.setattr(qbo_sale, "orders",
                        types.SimpleNamespace(set_order_sales_receipt_id=lambda *a, **k: None))
    out = qbo_sale.book_sale_on_payment(None, _order(qbo_sales_receipt_id="SRX"))
    assert out == "SRX"
    assert qb.receipts == 0


def test_best_effort_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("QBO down")
    monkeypatch.setattr(qbo_sale, "qbo_billing",
                        types.SimpleNamespace(find_or_create_customer=boom,
                                              create_sales_receipt=boom))
    monkeypatch.setattr(qbo_sale, "orders",
                        types.SimpleNamespace(set_order_sales_receipt_id=lambda *a, **k: None))
    assert qbo_sale.book_sale_on_payment(None, _order()) is None


def test_missing_lines_json_is_a_noop(monkeypatch):
    qb = _FakeQB()
    monkeypatch.setattr(qbo_sale, "qbo_billing", qb)
    monkeypatch.setattr(qbo_sale, "orders",
                        types.SimpleNamespace(set_order_sales_receipt_id=lambda *a, **k: None))
    assert qbo_sale.book_sale_on_payment(None, _order(qbo_lines_json=None)) is None
    assert qb.receipts == 0
