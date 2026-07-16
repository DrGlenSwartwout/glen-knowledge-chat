import importlib
import sys
import os
import pytest


def _load_app():
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app import failed: {e}")


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    app = _load_app()
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(app, "LOG_DB", db, raising=False)
    return app


def test_membership_books_sales_receipt_not_invoice(appmod, monkeypatch):
    from dashboard import qbo_billing

    calls = {"receipt": 0, "invoice": 0, "payment": 0}
    monkeypatch.setattr(qbo_billing, "find_or_create_customer",
                        lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(qbo_billing, "create_sales_receipt",
                        lambda *a, **k: calls.__setitem__("receipt", calls["receipt"] + 1)
                        or {"Id": "SR1"})
    monkeypatch.setattr(qbo_billing, "create_invoice",
                        lambda *a, **k: calls.__setitem__("invoice", calls["invoice"] + 1)
                        or {"Id": "INV1"})
    monkeypatch.setattr(qbo_billing, "record_payment",
                        lambda *a, **k: calls.__setitem__("payment", calls["payment"] + 1))

    appmod._book_membership_qbo("m@b.com", {"key": "month", "label": "1-Month",
                                         "price_cents": 9900})

    assert calls["receipt"] == 1
    assert calls["invoice"] == 0
    assert calls["payment"] == 0


def test_membership_qbo_failure_never_raises(appmod, monkeypatch):
    from dashboard import qbo_billing

    def boom(*a, **k):
        raise RuntimeError("QBO down")
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", boom)
    # Must swallow and log, not raise.
    appmod._book_membership_qbo("m@b.com", {"key": "month", "label": "1-Month",
                                         "price_cents": 9900})
