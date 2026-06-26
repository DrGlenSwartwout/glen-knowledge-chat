"""C2: finance.record_payment executor (mocked QBO) + the optional method memo."""
import pytest
from dashboard import finance, qbo_billing
from dashboard.actions import get_action, MONEY_SEND
from dashboard.rbac import OWNER, OPS


def test_record_payment_exec_calls_qbo(monkeypatch):
    calls = {}
    monkeypatch.setattr(qbo_billing, "get_invoice", lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "100"})
    def fake_rp(customer_id, amount_cents, invoice_id, method=None):
        calls.update(customer_id=customer_id, amount_cents=amount_cents, invoice_id=invoice_id, method=method)
        return {"Id": "P1"}
    monkeypatch.setattr(qbo_billing, "record_payment", fake_rp)
    res = finance._record_payment_exec({"invoice_id": "55", "amount": 50.0, "method": "Zelle"}, {})
    assert res["ok"] is True
    assert calls == {"customer_id": "42", "amount_cents": 5000, "invoice_id": "55", "method": "Zelle"}


def test_record_payment_exec_missing_invoice(monkeypatch):
    monkeypatch.setattr(qbo_billing, "get_invoice", lambda iid: None)
    res = finance._record_payment_exec({"invoice_id": "99", "amount": 10.0}, {})
    assert res.get("ok") is False


def test_record_payment_exec_rejects_nonpositive(monkeypatch):
    monkeypatch.setattr(qbo_billing, "get_invoice", lambda iid: {"CustomerRef": {"value": "1"}, "Balance": "10"})
    res = finance._record_payment_exec({"invoice_id": "1", "amount": 0}, {})
    assert res.get("ok") is False


def test_action_registered_metadata():
    a = get_action("finance.record_payment")
    assert a is not None and a.risk_tier == MONEY_SEND and a.permission == (OWNER, OPS)


def test_qbo_record_payment_method_memo(monkeypatch):
    cap = {}
    monkeypatch.setattr(qbo_billing, "get_invoice", lambda iid: {"Balance": "100"})
    def fake_post(path, body):
        cap["body"] = body
        return {"Payment": {"Id": "P1"}}
    monkeypatch.setattr(qbo_billing, "_post", fake_post)
    qbo_billing.record_payment("42", 5000, "55", method="Zelle")
    assert "Zelle" in cap["body"].get("PrivateNote", "")
    cap.clear()
    qbo_billing.record_payment("42", 5000, "55")
    assert "PrivateNote" not in cap["body"]
