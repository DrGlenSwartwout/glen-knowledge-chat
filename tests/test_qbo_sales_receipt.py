import pytest
from dashboard import qbo_billing as qb


@pytest.fixture
def captured(monkeypatch):
    """Capture the body posted to _post, stubbing all QBO I/O."""
    sink = {}

    def fake_post(path, body):
        sink["path"] = path
        sink["body"] = body
        return {"SalesReceipt": {"Id": "SR1", "DocNumber": "1001", "TotalAmt": body_total(body)}}

    def body_total(body):
        return round(sum(l["Amount"] for l in body.get("Line", [])), 2)

    monkeypatch.setattr(qb, "_post", fake_post)
    monkeypatch.setattr(qb, "_first_bank_account_id", lambda: "BANK9")
    monkeypatch.setattr(qb, "find_or_create_item", lambda name, price=None: {"Id": "IT1"})
    return sink


def test_posts_to_salesreceipt_with_deposit_account(captured):
    out = qb.create_sales_receipt({"Id": "C1"},
                                  [{"name": "Widget", "amount": 10.0, "qty": 2}])
    assert captured["path"] == "/salesreceipt"
    body = captured["body"]
    assert body["CustomerRef"] == {"value": "C1"}
    assert body["DepositToAccountRef"] == {"value": "BANK9"}
    line = body["Line"][0]
    assert line["DetailType"] == "SalesItemLineDetail"
    assert line["Amount"] == 20.0
    assert line["SalesItemLineDetail"]["ItemRef"] == {"value": "IT1"}
    assert out["Id"] == "SR1"


def test_resolves_provided_item_id_without_lookup(captured, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("find_or_create_item must not be called when item_id given")
    monkeypatch.setattr(qb, "find_or_create_item", boom)
    qb.create_sales_receipt({"Id": "C1"},
                            [{"name": "Widget", "amount": 5.0, "qty": 1, "item_id": "PRE"}])
    assert captured["body"]["Line"][0]["SalesItemLineDetail"]["ItemRef"] == {"value": "PRE"}


def test_tax_cents_stamps_totaltax_override(captured):
    qb.create_sales_receipt({"Id": "C1"}, [{"name": "W", "amount": 100.0, "qty": 1}],
                            tax_cents=475)
    body = captured["body"]
    assert body["TxnTaxDetail"] == {"TotalTax": 4.75}
    assert body["GlobalTaxCalculation"] == "TaxExcluded"


def test_zero_tax_omits_tax_detail(captured):
    qb.create_sales_receipt({"Id": "C1"}, [{"name": "W", "amount": 100.0, "qty": 1}])
    assert "TxnTaxDetail" not in captured["body"]


def test_discount_cents_appends_discount_line(captured):
    qb.create_sales_receipt({"Id": "C1"}, [{"name": "W", "amount": 100.0, "qty": 1}],
                            discount_cents=1500)
    disc = [l for l in captured["body"]["Line"] if l["DetailType"] == "DiscountLineDetail"]
    assert len(disc) == 1
    assert disc[0]["Amount"] == 15.0


def test_email_to_sets_billemail(captured):
    qb.create_sales_receipt({"Id": "C1"}, [{"name": "W", "amount": 1.0, "qty": 1}],
                            email_to="a@b.com")
    assert captured["body"]["BillEmail"] == {"Address": "a@b.com"}


def test_raises_without_bank_account(monkeypatch):
    monkeypatch.setattr(qb, "_first_bank_account_id", lambda: None)
    monkeypatch.setattr(qb, "find_or_create_item", lambda name, price=None: {"Id": "IT1"})
    with pytest.raises(RuntimeError, match="bank account"):
        qb.create_sales_receipt({"Id": "C1"}, [{"name": "W", "amount": 1.0, "qty": 1}])
