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
    from dashboard import stripe_pay, order_payments, qbo_billing

    calls = []

    def _counting_record_payment(*a, **k):
        calls.append((a, k))
        return {"Id": "P1"}

    monkeypatch.setattr(qbo_billing, "record_payment", _counting_record_payment)

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
    # The ledger (add_payment) owns the single QBO push for in-house orders;
    # the direct record_payment call below it in begin_checkout_return must
    # be gated off for kind="in-house" so it does not fire a second push.
    assert len(calls) == 1


def test_boot_creates_order_payments_table(tmp_path, monkeypatch):
    """The app's boot schema-init cluster must create order_payments itself —
    not rely on a lazy ensure_table() call from a route or test fixture. Seed
    ONLY the orders table (never call OP.ensure_table), reload app (boot),
    then confirm order_payments now exists in sqlite_master."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = str(tmp_path / "chat_log.db")
    from dashboard import orders as O
    with sqlite3.connect(db) as cx:
        O.init_orders_table(cx)
        cx.commit()

    repo = Path(__file__).resolve().parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import app as appmod
        importlib.reload(appmod)
    except Exception as e:
        pytest.skip(f"app not importable: {e}")

    cx = sqlite3.connect(db)
    row = cx.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='order_payments'").fetchone()
    cx.close()
    assert row is not None, "boot did not create order_payments table"


def test_client_invoice_shows_payments_and_balance(tmp_path, monkeypatch):
    """GET /api/invoice/<token> must surface the active-only payment ledger:
    a payments list, and balance_due_cents net of what's been paid. A voided
    payment must never appear in the payments list."""
    appmod, client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_bos_actor", lambda: {"role": "owner"})

    from dashboard import practitioner_portal as PP
    # The invoice-token lookup (_pp.order_id_from_invoice_token) uses PP's own
    # module-level db path by default — point it at the same chat_log.db the
    # _client fixture just seeded (order id=1), so the token resolves.
    db_path = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(PP, "_LOG_DB", Path(db_path))
    token = PP.create_order_invoice_token(1)

    # One active payment.
    pay = client.post("/api/orders/1/payments",
                       json={"amount": 131.00, "method": "Zelle"}).get_json()
    assert pay["ok"] is True

    # A second payment that gets voided — must be excluded from the client view.
    voided = client.post("/api/orders/1/payments",
                          json={"amount": 50.00, "method": "Cash"}).get_json()
    vpid = voided["row"]["id"]
    v = client.post(f"/api/orders/payments/{vpid}/void", json={"reason": "duplicate"})
    assert v.status_code == 200 and v.get_json()["row"]["status"] == "void"

    r = client.get(f"/api/invoice/{token}")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    order = body["order"]

    payments = order["payments"]
    assert len(payments) == 1, f"expected only the active payment, got {payments}"
    assert payments[0]["amount_cents"] == 13100
    assert payments[0]["kind"] == "payment"
    assert all(p["amount_cents"] != 5000 for p in payments), \
        "voided payment leaked into the client-facing payments list"

    # order total_cents=41282 (seeded by _client) minus the one active payment.
    assert order["balance_due_cents"] == 41282 - 13100
    assert order["refunded_cents"] == 0


def test_payments_list_includes_manual_payments(tmp_path, monkeypatch):
    """Zelle/check/cash payments recorded in order_payments (the manual ledger)
    must show up in GET /api/payments — the money view — alongside Stripe
    charges, not just live in the per-order ledger. A voided manual payment
    must never leak into the view."""
    appmod, client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(appmod, "_bos_actor", lambda: {"role": "owner"})

    pay = client.post("/api/orders/1/payments",
                       json={"amount": 75.00, "method": "Zelle"}).get_json()
    assert pay["ok"] is True

    # A second, voided payment must NOT leak into the money view.
    voided = client.post("/api/orders/1/payments",
                          json={"amount": 20.00, "method": "Cash"}).get_json()
    vpid = voided["row"]["id"]
    v = client.post(f"/api/orders/payments/{vpid}/void", json={"reason": "dup"})
    assert v.status_code == 200

    r = client.get("/api/payments")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True

    manual_rows = [row for row in body["data"]
                   if str(row.get("source", "")).startswith("manual:")]
    assert any(row["amount_cents"] == 7500 and "zelle" in row["source"].lower()
               for row in manual_rows), manual_rows
    assert all(row["amount_cents"] != 2000 for row in manual_rows), \
        "voided manual payment leaked into /api/payments"
