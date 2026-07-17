"""QBO paid-only Stage 4, Task 1 gap fix: /practitioner/checkout-return.

Wholesale/dispensary CARD payments return to /practitioner/checkout-return
(never /begin/checkout-return). Stage 4 Task 1 converted those orders to
paid-only (qbo_lines_json on the order, Stripe metadata customer_id=""), but
the return handler's only work lived inside `if inv and cid:` -- skipped
whenever cid=="" (every paid-only order). Net effect: a paid-only
wholesale/dispensary card payment never got marked paid and never booked a
Sales Receipt. (Alt-pay was fine -- it goes through _record_payment_exec ->
book_sale_on_payment.)

These tests cover the new paid-only branch added after the legacy
`if inv and cid:` block:
  - a paid-only order (qbo_lines_json set, cid=="") gets pay_status=paid,
    the Stripe PI captured, and exactly one Sales Receipt booked;
  - re-hitting the same return URL books no second receipt (idempotent).

(QBO Stage 5: the legacy `if inv and cid:` record_payment block itself was
removed as dead code -- it only ever fired for legacy invoice-based orders,
which are voided/drained and never occur in practice. The test that locked
in that legacy behavior was removed along with it.)
"""

import sqlite3

import pytest

import app
from dashboard import orders as O
from dashboard import qbo_billing
from dashboard import stripe_pay


def _isolate_db(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    cx = sqlite3.connect(db)
    try:
        O.init_orders_table(cx)
        cx.commit()
    finally:
        cx.close()
    return db


def _seed_paid_only_order(db, token, *, total_cents=50000):
    cx = sqlite3.connect(db)
    try:
        oid = O.upsert_order(cx, source="wholesale", external_ref=token,
                              email="dr@x.com", name="Dr X", total_cents=total_cents,
                              channel="wholesale", practitioner_id="pid1")
        O.set_order_qbo_lines(cx, token, {
            "lines": [{"name": "X Formula", "amount": 25.0, "qty": 20, "item_id": "55"}],
            "discount_cents": 0, "tax_cents": 0,
        })
        cx.commit()
    finally:
        cx.close()
    return oid


@pytest.fixture
def client():
    app.app.config["TESTING"] = True
    return app.app.test_client()


def test_paid_only_wholesale_card_return_marks_paid_and_books_one_receipt(
        monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    token = "a" * 32
    _seed_paid_only_order(db, token)

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "id": sid, "payment_status": "paid", "amount_total": 50000,
        "metadata": {"invoice_id": token, "customer_id": ""},
        "payment_intent": "pi_123",
    })

    calls = {"n": 0}

    def _fake_create_sales_receipt(customer, lines, *, discount_cents=0, tax_cents=0,
                                    email_to=None):
        calls["n"] += 1
        return {"Id": "SR1"}

    monkeypatch.setattr(qbo_billing, "find_or_create_customer",
                        lambda email, name="": {"Id": "C9"})
    monkeypatch.setattr(qbo_billing, "create_sales_receipt", _fake_create_sales_receipt)

    def boom(*a, **k):
        raise AssertionError("legacy record_payment must not run for a paid-only order")
    monkeypatch.setattr(qbo_billing, "record_payment", boom)

    r = client.get(f"/practitioner/checkout-return?session_id=sess1&t={token}")
    assert r.status_code in (301, 302)

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row["pay_status"] == "paid"
    assert row["stripe_payment_intent"] == "pi_123"
    assert row["qbo_sales_receipt_id"] == "SR1"
    assert calls["n"] == 1


def test_paid_only_wholesale_card_return_rehit_is_idempotent(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    token = "b" * 32
    _seed_paid_only_order(db, token)

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "id": sid, "payment_status": "paid", "amount_total": 50000,
        "metadata": {"invoice_id": token, "customer_id": ""},
        "payment_intent": "pi_456",
    })

    calls = {"n": 0}

    def _fake_create_sales_receipt(customer, lines, *, discount_cents=0, tax_cents=0,
                                    email_to=None):
        calls["n"] += 1
        return {"Id": "SR2"}

    monkeypatch.setattr(qbo_billing, "find_or_create_customer",
                        lambda email, name="": {"Id": "C9"})
    monkeypatch.setattr(qbo_billing, "create_sales_receipt", _fake_create_sales_receipt)

    client.get(f"/practitioner/checkout-return?session_id=sess1&t={token}")
    client.get(f"/practitioner/checkout-return?session_id=sess1&t={token}")

    assert calls["n"] == 1

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row["qbo_sales_receipt_id"] == "SR2"
