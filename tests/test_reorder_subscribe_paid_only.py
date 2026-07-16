"""Task 4: reorder_subscribe (kind=='subscribe') -> QBO paid-only.

Mirrors tests/test_checkout_cart_paid_only.py's DB-isolation fixture. Confirms:
(a) the guard -- no QBO invoice is ever created, the checkout mints a token
    instead of a real invoice id;
(b) the order is keyed on that token, with the exact qbo_lines_json payload
    persisted for later Sales-Receipt booking;
(c) the SEPARATE /begin/checkout-return "subscribe" block (which vaults the
    card and writes the subscription row) is unaffected -- it keys off
    Stripe metadata (items/ship/cadence), not the invoice_id, so it must
    still fire correctly once invoice_id is a token rather than a real QBO
    invoice id.
"""
import json
import sqlite3

import app
from dashboard import orders as O
from dashboard import qbo_billing


def _client():
    return app.app.test_client()


def _isolate_db(monkeypatch, tmp_path):
    """Point app.LOG_DB at a throwaway sqlite file (never touching the real local
    chat_log.db) and init the orders schema."""
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    cx = sqlite3.connect(db)
    try:
        O.init_orders_table(cx)
        cx.commit()
    finally:
        cx.close()
    return db


PRODUCT_SLUG = "brain-boost"


def _prep(monkeypatch, tmp_path, email="sub@x.com"):
    db = _isolate_db(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise AssertionError("reorder_subscribe must not call create_invoice (paid-only)")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})

    monkeypatch.setattr(app, "_reorder_email_from_cookie", lambda: email)
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(
        app, "_get_product",
        lambda s: {"slug": s, "name": "Brain Boost", "price_cents": 7000,
                   "qty_pricing": True, "qbo_item_id": "27"} if s == PRODUCT_SLUG else None)
    monkeypatch.setattr(app._shipping, "quote", lambda b: {"shipping_cents": 2295})

    monkeypatch.setenv("PRICING_ENGINE_CHECKOUT", "true")
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True)

    cap = {}
    import dashboard.stripe_pay as sp
    monkeypatch.setattr(
        sp, "create_checkout_session",
        lambda *a, **k: cap.update(k) or {"id": "cs_1", "url": "https://stripe/setup"})
    return db, cap


def _post_subscribe():
    return _client().post(
        "/reorder/subscribe",
        json={"items": [{"slug": PRODUCT_SLUG, "qty": 1}], "cadence_months": 1,
              "address": {"state": "CA", "country": "US", "name": "A"}})


def test_subscribe_creates_no_qbo_invoice_and_mints_token(monkeypatch, tmp_path):
    """Guard: reorder_subscribe must NOT create a QBO invoice; metadata carries a
    token (not a real invoice id) as invoice_id, and customer_id is empty."""
    _prep(monkeypatch, tmp_path)
    r = _post_subscribe()
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert body["stripe_url"] == "https://stripe/setup"


def test_subscribe_metadata_invoice_id_is_token_customer_id_empty(monkeypatch, tmp_path):
    db, cap = _prep(monkeypatch, tmp_path)
    r = _post_subscribe()
    assert r.status_code == 200, r.get_data(as_text=True)
    md = cap["metadata"]
    assert md["kind"] == "subscribe"
    assert md["cadence_months"] == "1"
    assert md["customer_id"] == ""
    token = md["invoice_id"]
    assert token  # non-empty
    # a real QBO invoice id from this app is numeric-ish/short; the token is a
    # full uuid4 hex (32 chars) -- confirms it's the checkout_ref, not an invoice.
    assert len(token) == 32
    int(token, 16)  # parses as hex -- proves it's uuid4().hex, not a QBO Id


def test_subscribe_order_keyed_on_token_with_qbo_lines_persisted(monkeypatch, tmp_path):
    db, cap = _prep(monkeypatch, tmp_path)
    r = _post_subscribe()
    assert r.status_code == 200, r.get_data(as_text=True)
    token = cap["metadata"]["invoice_id"]

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row is not None
    assert row["source"] == "subscribe"
    payload = json.loads(row["qbo_lines_json"])
    assert payload["lines"]  # line-faithful payload persisted
    # the shipping line rides along in the same lines list (as create_invoice used to get)
    assert any(l.get("name") == "Shipping (USPS)" for l in payload["lines"])


def test_subscribe_return_still_writes_subscription_row(monkeypatch, tmp_path):
    """PINNING test: the dedicated 'subscribe' return-handler block (app.py ~9479)
    keys off Stripe metadata (items/ship/cadence/email), not the invoice_id -- so
    it must still vault the card and create the subscription row even though
    invoice_id is now a paid-only checkout token, not a real QBO invoice id."""
    db = _isolate_db(monkeypatch, tmp_path)
    token = "chk_pin_subscribe_1"
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    try:
        O.upsert_order(cx, source="subscribe", external_ref=token, email="sub2@x.com",
                       name="Sub Two", items=[{"slug": PRODUCT_SLUG, "qty": 1}],
                       total_cents=7000, address={}, channel="retail",
                       get_cents=0, discount_cents=0, points_redeemed_cents=0,
                       shipping_cents=2295, status="new")
        O.set_order_qbo_lines(cx, token, {
            "lines": [{"name": "Brain Boost", "amount": 70.0, "qty": 1},
                      {"name": "Shipping (USPS)", "amount": 22.95, "qty": 1}],
            "discount_cents": 0, "tax_cents": 0})
    finally:
        cx.close()

    from dashboard import subscriptions as subs_mod
    calls = {"create": []}
    _real_create = subs_mod.create

    def spy_create(cx2, **kw):
        calls["create"].append(kw)
        return _real_create(cx2, **kw)
    monkeypatch.setattr(subs_mod, "create", spy_create)

    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 9295, "payment_intent": "pi_sub_1",
        "metadata": {
            "kind": "subscribe", "cadence_months": "1", "email": "sub2@x.com",
            "invoice_id": token, "customer_id": "",
            "items": json.dumps([{"slug": PRODUCT_SLUG, "qty": 1}]),
            "ship": json.dumps({"state": "CA", "country": "US", "name": "Sub Two"}),
        }})
    monkeypatch.setattr(sp, "get_payment_intent", lambda pi_id: {
        "customer": "cus_sub_1", "payment_method": "pm_sub_1"})

    r = _client().get("/begin/checkout-return?session_id=sess_sub_1")
    assert r.status_code in (301, 302)

    assert len(calls["create"]) == 1
    created = calls["create"][0]
    assert created["email"] == "sub2@x.com"
    assert created["stripe_customer_id"] == "cus_sub_1"
    assert created["stripe_payment_method_id"] == "pm_sub_1"
    assert created["cadence_months"] == 1
