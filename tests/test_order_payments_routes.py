import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard import orders as O
    from dashboard import order_payments as OP
    with sqlite3.connect(db) as cx:
        O.init_orders_table(cx)
        OP.ensure_table(cx)
        O.upsert_order(cx, source="qbo", external_ref="INV-1",
                        email="d@e.com", total_cents=41282)
        cx.commit()
    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")
    from dashboard import qbo_billing
    monkeypatch.setattr(qbo_billing, "get_invoice",
                         lambda iid: {"CustomerRef": {"value": "42"}, "Balance": "9"})
    monkeypatch.setattr(qbo_billing, "record_payment", lambda *a, **k: {"Id": "P1"})
    appmod.app.config["TESTING"] = True
    return appmod, appmod.app.test_client()


def test_add_payment_requires_actor(tmp_path, monkeypatch):
    appmod, client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_bos_actor", lambda: None)
    r = client.post("/api/orders/1/payments", json={"amount": 131, "method": "Zelle"})
    assert r.status_code == 401


def test_add_payment_and_balance(tmp_path, monkeypatch):
    appmod, client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_bos_actor", lambda: {"role": "owner"})
    r = client.post("/api/orders/1/payments", json={"amount": 131.00, "method": "Zelle"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    g = client.get("/api/orders/1/payments").get_json()
    assert g["balance"]["paid_cents"] == 13100


def test_add_refund_and_void_and_resync(tmp_path, monkeypatch):
    appmod, client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_bos_actor", lambda: {"role": "owner"})
    pay = client.post("/api/orders/1/payments",
                       json={"amount": 131.00, "method": "Zelle"}).get_json()
    pid = pay["row"]["id"]

    r = client.post("/api/orders/1/refunds", json={"amount": 31.00, "method": "Zelle"})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert r.get_json()["balance"]["refunded_cents"] == 3100

    v = client.post(f"/api/orders/payments/{pid}/void", json={"reason": "duplicate"})
    assert v.status_code == 200
    assert v.get_json()["row"]["status"] == "void"

    rs = client.post(f"/api/orders/payments/{pid}/resync")
    assert rs.status_code == 200 and rs.get_json()["ok"] is True


def test_checkout_return_creates_one_stripe_row(tmp_path, monkeypatch):
    appmod, client = _client(tmp_path, monkeypatch)
    from dashboard import stripe_pay, order_payments

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "payment_intent": "pi_777",
        "amount_total": 22291,
        "metadata": {"kind": "in-house", "invoice_id": "INV-1", "customer_id": "42"}})

    client.get("/begin/checkout-return?kind=in-house&session_id=cs_1")
    client.get("/begin/checkout-return?kind=in-house&session_id=cs_1")  # retry

    cx = sqlite3.connect(appmod.LOG_DB)
    cx.row_factory = sqlite3.Row
    order_payments.ensure_table(cx)
    oid = cx.execute(
        "SELECT id FROM orders WHERE external_ref='INV-1'").fetchone()[0]
    rows = [r for r in order_payments.list_payments(cx, oid)
            if r["kind"] == "payment" and r["source"] == "stripe"]
    cx.close()
    assert len(rows) == 1 and rows[0]["amount_cents"] == 22291
