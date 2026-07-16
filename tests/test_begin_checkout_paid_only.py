import json
import sqlite3

import app
from dashboard import orders as O
from dashboard import qbo_billing


def _client():
    return app.app.test_client()


def _isolate_db(monkeypatch, tmp_path):
    """Point app.LOG_DB at a throwaway sqlite file (never touching the real local
    chat_log.db) and init the orders schema, mirroring
    tests/test_biofield_checkout_paid_only.py's _isolate_db fixture (Task 3) --
    these tests don't stub out _ingest_order, so the orders table must actually
    exist on the fresh db."""
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


def _prep(monkeypatch, tmp_path, method="card"):
    """Isolate the DB, stub the product catalog with a real-shaped entry (mirrors
    tests/test_begin_checkout_engine.py's _setup fixture), guard against any QBO
    invoice write, and stub Stripe session creation so no network call happens."""
    db = _isolate_db(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise AssertionError("begin_checkout must not call create_invoice (paid-only)")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})

    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(
        app, "_get_product",
        lambda s: {"slug": s, "name": "Brain Boost", "price_cents": 7000,
                   "qty_pricing": True, "qbo_item_id": "27"} if s == PRODUCT_SLUG else None)
    monkeypatch.setattr(app._shipping, "quote", lambda b: {"shipping_cents": 2295})

    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "create_checkout_session", lambda *a, **k: {"url": "https://s.test"})
    return db


def test_begin_checkout_creates_no_qbo_invoice(monkeypatch, tmp_path):
    """Guard (mutation-style): begin_checkout must NOT POST an invoice to QBO."""
    _prep(monkeypatch, tmp_path)
    r = _client().post(f"/begin/checkout/{PRODUCT_SLUG}",
                       json={"email": "c@b.com", "name": "C", "qty": 1,
                             "method": "card",
                             "address": {"state": "CA", "country": "US", "name": "C"}})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert body.get("invoice_id")  # token under the compat field
    assert body.get("customer_id") == ""


def test_begin_checkout_charges_subtotal_plus_shipping_not_get(monkeypatch, tmp_path):
    """Money-path fix: the Stripe charge (and the stored order total_cents) must equal
    subtotal + shipping, NOT subtotal + GET. Real HI ship-to + TAX_ENABLED so the
    pricing engine computes a genuine nonzero get_cents -- before the fix this GET
    leaked into the charged amount while shipping was dropped entirely."""
    db = _prep(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True)
    monkeypatch.setenv("TAX_ENABLED", "true")
    monkeypatch.setenv("GET_RETAIL_RATE", "0.045")

    cap = {}
    import dashboard.stripe_pay as sp
    def fake_session(amount_cents, **kw):
        cap["amount_cents"] = amount_cents
        return {"url": "https://s.test"}
    monkeypatch.setattr(sp, "create_checkout_session", fake_session)

    r = _client().post(f"/begin/checkout/{PRODUCT_SLUG}",
                       json={"email": "hi@b.com", "name": "H", "qty": 1,
                             "method": "card",
                             "address": {"state": "HI", "country": "US", "name": "H",
                                         "street": "1 Aloha", "city": "Hilo", "zip": "96720"}})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    # subtotal 7000 (qty 1, no discount) + shipping stub 2295 = 9295. GET (315) must
    # NOT be charged; shipping (previously dropped entirely) must be included.
    assert cap["amount_cents"] == 9295
    assert body["total"] == 92.95

    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, body["invoice_id"])
    assert row["total_cents"] == 9295
    assert row["get_cents"] == 315            # still recorded on the order for GET remittance
    assert row["shipping_cents"] == 2295

    payload = json.loads(row["qbo_lines_json"])
    receipt_total_cents = round(sum(l["amount"] for l in payload["lines"]) * 100) - payload["discount_cents"]
    assert receipt_total_cents == 9295        # charge matches the booked Sales Receipt


def test_begin_checkout_applies_ship_credit_to_charge_not_just_receipt(monkeypatch, tmp_path):
    """Money-path fix: a ship-credit balance auto-applied at checkout must reduce the
    Stripe CHARGE, not just the QBO Sales-Receipt discount. Policy: charge == booked
    receipt total == subtotal + shipping - ship_credit (floored at 0)."""
    db = _prep(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True)
    monkeypatch.setenv("SHIP_CREDIT_AUTOAPPLY_ENABLED", "1")

    from dashboard import points as P
    from dashboard import ship_credit as SC
    email = "sc@b.com"
    cx = sqlite3.connect(db)
    P.init_points_table(cx)
    SC.grant(cx, email, 500, source_ref="SRC-1")
    cx.close()

    cap = {}
    import dashboard.stripe_pay as sp
    def fake_session(amount_cents, **kw):
        cap["amount_cents"] = amount_cents
        return {"url": "https://s.test"}
    monkeypatch.setattr(sp, "create_checkout_session", fake_session)

    r = _client().post(f"/begin/checkout/{PRODUCT_SLUG}",
                       json={"email": email, "name": "S", "qty": 1,
                             "method": "card",
                             "address": {"state": "CA", "country": "US", "name": "S"}})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    # subtotal 7000 + shipping stub 2295 = 9295; minus 500 ship credit = 8795.
    assert cap["amount_cents"] == 8795
    assert body["total"] == 87.95

    cx2 = sqlite3.connect(db); cx2.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx2, body["invoice_id"])
    assert row["total_cents"] == 8795
    assert row["ship_credit_applied_cents"] == 500

    payload = json.loads(row["qbo_lines_json"])
    receipt_total_cents = round(sum(l["amount"] for l in payload["lines"]) * 100) - payload["discount_cents"]
    assert receipt_total_cents == 8795  # charge matches the booked Sales Receipt


def test_begin_checkout_ship_credit_floors_charge_at_zero(monkeypatch, tmp_path):
    """A ship-credit balance larger than the order total must floor the charge (and
    stored total) at 0, never go negative."""
    db = _prep(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True)
    monkeypatch.setenv("SHIP_CREDIT_AUTOAPPLY_ENABLED", "1")

    from dashboard import points as P
    from dashboard import ship_credit as SC
    email = "bigcredit@b.com"
    cx = sqlite3.connect(db)
    P.init_points_table(cx)
    SC.grant(cx, email, 100000, source_ref="SRC-BIG")
    cx.close()

    cap = {}
    import dashboard.stripe_pay as sp
    def fake_session(amount_cents, **kw):
        cap["amount_cents"] = amount_cents
        return {"url": "https://s.test"}
    monkeypatch.setattr(sp, "create_checkout_session", fake_session)

    r = _client().post(f"/begin/checkout/{PRODUCT_SLUG}",
                       json={"email": email, "name": "B", "qty": 1,
                             "method": "card",
                             "address": {"state": "CA", "country": "US", "name": "B"}})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["total"] == 0.0
    # ship_credit.plan_application bounds the applied credit to the order's own
    # chargeable total (9295), so the ledger records only what was actually usable.
    cx2 = sqlite3.connect(db); cx2.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx2, body["invoice_id"])
    assert row["total_cents"] == 0
    assert row["ship_credit_applied_cents"] == 9295


def test_begin_checkout_persists_qbo_lines_and_token_ref(monkeypatch, tmp_path):
    db = _prep(monkeypatch, tmp_path)
    r = _client().post(f"/begin/checkout/{PRODUCT_SLUG}",
                       json={"email": "d@b.com", "name": "D", "qty": 1,
                             "method": "card",
                             "address": {"state": "CA", "country": "US", "name": "D"}})
    ref = r.get_json()["invoice_id"]
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, ref)
    assert row is not None and row["source"] == "funnel"
    payload = json.loads(row["qbo_lines_json"])
    assert payload["lines"]  # line-faithful payload was stored


def test_funnel_stripe_return_settles_pi_points_referral_and_books_once(monkeypatch, tmp_path):
    """PINNING test (closes the gap the Task 4 brief flagged): a funnel Stripe-return
    with a non-empty payment-intent must still stamp the PaymentIntent, settle points,
    settle the referral, and book exactly ONE QBO Sales Receipt -- even though the
    checkout route now sets Stripe metadata customer_id="" (paid-only), which would
    silently kill all four side effects if the return-handler gate still required a
    truthy `cid` alongside `inv`."""
    db = _isolate_db(monkeypatch, tmp_path)
    token = "chk_pin_test_1"
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    try:
        oid = O.upsert_order(cx, source="funnel", external_ref=token, email="pin@b.com",
                             name="Pin", items=[{"slug": PRODUCT_SLUG, "qty": 1}],
                             total_cents=7000, address={}, channel="retail",
                             get_cents=0, discount_cents=0, points_redeemed_cents=0,
                             shipping_cents=0, status="new")
        O.set_order_qbo_lines(cx, token, {
            "lines": [{"name": "Brain Boost", "amount": 70.0, "qty": 1}],
            "discount_cents": 0, "tax_cents": 0})
    finally:
        cx.close()

    calls = {"pi": [], "points": [], "referral": [], "booked": []}
    monkeypatch.setattr(app, "_settle_order_points",
                        lambda order, *, order_ref: calls["points"].append(order_ref))
    monkeypatch.setattr(app, "_settle_referral",
                        lambda order, *, order_ref: calls["referral"].append(order_ref))

    orig_set_pi = app._bos_orders.set_order_stripe_pi

    def spy_set_pi(cx2, order_id, pi):
        calls["pi"].append((order_id, pi))
        return orig_set_pi(cx2, order_id, pi)
    monkeypatch.setattr(app._bos_orders, "set_order_stripe_pi", spy_set_pi)

    import dashboard.qbo_sale as _qs
    orig_book = _qs.book_sale_on_payment

    def spy_book(cx2, order):
        calls["booked"].append(order.get("external_ref"))
        return orig_book(cx2, order)
    monkeypatch.setattr(_qs, "book_sale_on_payment", spy_book)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(qbo_billing, "create_sales_receipt", lambda *a, **k: {"Id": "SR1"})

    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 7000, "payment_intent": "pi_123",
        "metadata": {"kind": "retail", "invoice_id": token, "customer_id": "",
                     "slug": PRODUCT_SLUG}})

    r = _client().get("/begin/checkout-return?session_id=sess1")
    assert r.status_code in (301, 302)

    assert calls["pi"] == [(oid, "pi_123")]
    assert calls["points"] == [token]
    assert calls["referral"] == [token]
    assert calls["booked"] == [token]  # exactly one booking call

    cx2 = sqlite3.connect(db); cx2.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx2, token)
    assert row["qbo_sales_receipt_id"] == "SR1"

    # The generic invoice-apply (record_payment) path must NEVER fire for a
    # paid-only funnel/retail order -- it has no real QBO customer/invoice to apply to.
    def boom(*a, **k):
        raise AssertionError("record_payment must not be called for kind=='retail'")
    monkeypatch.setattr(qbo_billing, "record_payment", boom)
    r2 = _client().get("/begin/checkout-return?session_id=sess1")
    assert r2.status_code in (301, 302)
    # second pass is idempotent: no second booking call (existing receipt short-circuits)
    assert calls["booked"] == [token, token]
    cx3 = sqlite3.connect(db); cx3.row_factory = sqlite3.Row
    row3 = O.find_order_by_external_ref(cx3, token)
    assert row3["qbo_sales_receipt_id"] == "SR1"
