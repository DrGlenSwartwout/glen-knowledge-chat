"""QBO webhook-back booking: /webhook/stripe books the paid-only Sales Receipt
directly on checkout.session.completed, so a closed browser tab (dropped
redirect) can't leave money collected with no QBO receipt + an order stuck
unpaid. Mirrors the /practitioner/checkout-return paid-only branch
(app.py:~25845-25868), but fired from the webhook instead of the return page.

Guarded on qbo_lines_json (paid-only checkout orders only -- trials and
memberships have no qbo_lines_json and are untouched) and idempotent via
book_sale_on_payment's atomic claim (qbo_sales_receipt_id), so it can never
double-book against the redirect handler. Best-effort: any failure inside the
block must be swallowed and the webhook must still 200.
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


def _seed_paid_only_order(db, token, *, total_cents=70000, with_lines=True,
                           already_booked=False):
    cx = sqlite3.connect(db)
    try:
        oid = O.upsert_order(cx, source="wholesale", external_ref=token,
                              email="dr@x.com", name="Dr X", total_cents=total_cents,
                              channel="wholesale", practitioner_id="pid1")
        if with_lines:
            O.set_order_qbo_lines(cx, token, {
                "lines": [{"name": "X Formula", "amount": 25.0, "qty": 20, "item_id": "55"}],
                "discount_cents": 0, "tax_cents": 0,
            })
        if already_booked:
            O.claim_sales_receipt_slot(cx, oid)
            O.set_order_sales_receipt_id(cx, oid, "SR-EXISTING")
        cx.commit()
    finally:
        cx.close()
    return oid


def _noop_fulfillers(monkeypatch):
    # Isolate the new webhook-back-booking block from the pre-existing
    # fulfillers (each independently re-fetches the session and would
    # otherwise also run against our mocked get_session).
    for name in ("_fulfill_biofield_trial", "_fulfill_prepay_term",
                 "_fulfill_biofield_program", "_fulfill_continuous_care_monthly",
                 "_fulfill_membership_product", "_fulfill_masterclass",
                 "_fulfill_coach_sub", "_fulfill_family_plan"):
        monkeypatch.setattr(app, name, lambda sid: None)


def _event(session_id="cs_evt"):
    import json
    return json.dumps({"type": "checkout.session.completed",
                        "data": {"object": {"id": session_id}}}).encode()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    app.app.config["TESTING"] = True
    return app.app.test_client()


def _mock_sales_receipt_spy(monkeypatch):
    calls = {"n": 0}

    def _fake_create_sales_receipt(customer, lines, *, discount_cents=0, tax_cents=0,
                                    email_to=None):
        calls["n"] += 1
        return {"Id": f"SR{calls['n']}"}

    monkeypatch.setattr(qbo_billing, "find_or_create_customer",
                        lambda email, name="": {"Id": "C9"})
    monkeypatch.setattr(qbo_billing, "create_sales_receipt", _fake_create_sales_receipt)
    return calls


def test_webhook_books_paid_only_order_when_redirect_missed(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "a" * 32
    _seed_paid_only_order(db, token)

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 70000,
        "metadata": {"invoice_id": token}, "payment_intent": "pi_1"})
    calls = _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row["pay_status"] == "paid"
    assert row["stripe_payment_intent"] == "pi_1"
    assert row["qbo_sales_receipt_id"] == "SR1"
    assert calls["n"] == 1


def test_webhook_noop_when_already_booked(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "b" * 32
    _seed_paid_only_order(db, token, already_booked=True)

    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 70000,
        "metadata": {"invoice_id": token}, "payment_intent": "pi_2"})
    calls = _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200
    assert calls["n"] == 0

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row["qbo_sales_receipt_id"] == "SR-EXISTING"
    # book_sale_on_payment's guard (qbo_sales_receipt_id already set) fires before
    # set_order_payment runs, so an already-booked order's pay_status is untouched.
    assert row["pay_status"] == "unpaid"


def test_webhook_noop_for_non_paidonly_session(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "c" * 32
    # No matching order at all for this token (e.g. a trial/membership session
    # with no qbo_lines_json order ever created).
    monkeypatch.setattr(stripe_pay, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 100,
        "metadata": {"invoice_id": token}, "payment_intent": "pi_3"})
    calls = _mock_sales_receipt_spy(monkeypatch)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200
    assert calls["n"] == 0

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row is None


def test_webhook_swallows_booking_error_returns_200(monkeypatch, tmp_path, client):
    db = _isolate_db(monkeypatch, tmp_path)
    _noop_fulfillers(monkeypatch)
    token = "d" * 32
    _seed_paid_only_order(db, token)

    def _boom(sid):
        raise RuntimeError("stripe API blew up")
    monkeypatch.setattr(stripe_pay, "get_session", _boom)

    r = client.post("/webhook/stripe", data=_event(), content_type="application/json")
    assert r.status_code == 200
