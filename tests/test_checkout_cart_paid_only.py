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
    tests/test_begin_checkout_paid_only.py's _isolate_db fixture -- these tests
    don't stub out _ingest_order, so the orders table must actually exist on the
    fresh db."""
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


def _prep(monkeypatch, tmp_path, email="a@x.com"):
    """Isolate the DB, stub the product catalog with a real-shaped entry (mirrors
    tests/test_reorder_checkout_engine.py's _setup fixture), guard against any QBO
    invoice write, and stub Stripe session creation so no network call happens."""
    db = _isolate_db(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise AssertionError("_checkout_cart must not call create_invoice (paid-only)")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})

    monkeypatch.setattr(app, "_reorder_email_from_cookie", lambda: email)
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(
        app, "_get_product",
        lambda s: {"slug": s, "name": "Brain Boost", "price_cents": 7000,
                   "qty_pricing": True, "qbo_item_id": "27"} if s == PRODUCT_SLUG else None)
    monkeypatch.setattr(app._shipping, "quote", lambda b: {"shipping_cents": 2295})

    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "create_checkout_session", lambda *a, **k: {"url": "https://s.test"})
    return db


def test_checkout_cart_creates_no_qbo_invoice(monkeypatch, tmp_path):
    """Guard (mutation-style): _checkout_cart (reorder) must NOT POST an invoice to QBO."""
    _prep(monkeypatch, tmp_path)
    r = _client().post("/reorder/checkout",
                       json={"items": [{"slug": PRODUCT_SLUG, "qty": 6}],
                             "address": {"state": "CA", "country": "US", "name": "A"}})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert body.get("invoice_id")   # token under the compat field
    assert body.get("customer_id") == ""


def test_checkout_cart_persists_qbo_lines_and_token_ref(monkeypatch, tmp_path):
    db = _prep(monkeypatch, tmp_path)
    r = _client().post("/reorder/checkout",
                       json={"items": [{"slug": PRODUCT_SLUG, "qty": 6}],
                             "address": {"state": "CA", "country": "US", "name": "A"}})
    ref = r.get_json()["invoice_id"]
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, ref)
    assert row is not None and row["source"] == "reorder"
    payload = json.loads(row["qbo_lines_json"])
    assert payload["lines"]  # line-faithful payload was stored


def test_checkout_cart_charges_subtotal_plus_shipping_not_get(monkeypatch, tmp_path):
    """Policy (2026-07-16 design spec): GET is absorbed/recorded, never charged;
    shipping IS charged. So the charged amount (out["total"]) and the stored
    order total_cents must equal subtotal + shipping -- NOT subtotal + GET
    (pc["priced"]["total_cents"]) -- and must match the qbo_lines_json payload
    total (subtotal + shipping, tax_cents 0)."""
    db = _prep(monkeypatch, tmp_path)
    # Force a non-zero GET so a bug that charges pc["priced"]["total_cents"]
    # (subtotal + GET) is distinguishable from the correct subtotal + shipping.
    import dashboard.tax as _tax_mod
    monkeypatch.setattr(_tax_mod, "compute_get_cents", lambda *a, **k: 189)

    # Spy on _price_cart to capture its actual (post-discount) priced dict, so the
    # expected amount doesn't have to duplicate the volume-discount math.
    captured = {}
    orig_price_cart = app._price_cart

    def spy_price_cart(*a, **k):
        pc = orig_price_cart(*a, **k)
        captured["pc"] = pc
        return pc
    monkeypatch.setattr(app, "_price_cart", spy_price_cart)

    r = _client().post("/reorder/checkout",
                       json={"items": [{"slug": PRODUCT_SLUG, "qty": 6}],
                             "address": {"state": "HI", "country": "US", "name": "A"}})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True

    pc = captured["pc"]
    get_cents = int(pc["priced"]["get_cents"])
    shipping_cents = int(pc["shipping_cents"])
    assert get_cents == 189  # confirms GET really was non-zero for this cart
    assert shipping_cents == 2295
    expected_cents = int(pc["priced"]["total_cents"]) - get_cents + shipping_cents

    charged_cents = int(round(float(body["total"]) * 100))
    assert charged_cents == expected_cents

    ref = body["invoice_id"]
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, ref)
    assert int(row["total_cents"]) == expected_cents
    assert int(row["get_cents"]) == 189  # still recorded (absorbed) for remittance

    payload = json.loads(row["qbo_lines_json"])
    # Lines carry LIST price; the payload's own discount_cents (which does NOT
    # include GET -- GET isn't on this payload at all) is applied at the
    # receipt level, same as the real Sales-Receipt booking.
    list_lines_cents = sum(round(l["amount"] * 100) * l.get("qty", 1) for l in payload["lines"])
    qbo_total_cents = list_lines_cents - int(payload["discount_cents"])
    assert qbo_total_cents == expected_cents


def test_reorder_stripe_return_settles_pi_points_referral_and_books_once(monkeypatch, tmp_path):
    """PINNING test: a reorder Stripe-return with a non-empty payment-intent must
    still stamp the PaymentIntent, settle points, settle the referral, and book
    exactly ONE QBO Sales Receipt -- even though _checkout_cart now sets Stripe
    metadata customer_id="" (paid-only), which would silently kill all four side
    effects if the return-handler gate still required a truthy `cid` alongside
    `inv` (kind not in the extended kind-lists)."""
    db = _isolate_db(monkeypatch, tmp_path)
    token = "chk_pin_reorder_1"
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    try:
        oid = O.upsert_order(cx, source="reorder", external_ref=token, email="pin@b.com",
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
        "payment_status": "paid", "amount_total": 7000, "payment_intent": "pi_456",
        "metadata": {"kind": "reorder", "invoice_id": token, "customer_id": ""}})

    r = _client().get("/begin/checkout-return?session_id=sess1")
    assert r.status_code in (301, 302)

    assert calls["pi"] == [(oid, "pi_456")]
    assert calls["points"] == [token]
    assert calls["referral"] == [token]
    assert calls["booked"] == [token]  # exactly one booking call

    cx2 = sqlite3.connect(db); cx2.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx2, token)
    assert row["qbo_sales_receipt_id"] == "SR1"

    # The generic invoice-apply (record_payment) path must NEVER fire for a
    # paid-only reorder order -- it has no real QBO customer/invoice to apply to.
    def boom(*a, **k):
        raise AssertionError("record_payment must not be called for kind=='reorder'")
    monkeypatch.setattr(qbo_billing, "record_payment", boom)
    r2 = _client().get("/begin/checkout-return?session_id=sess1")
    assert r2.status_code in (301, 302)
    # second pass is idempotent: no second booking call (existing receipt short-circuits)
    assert calls["booked"] == [token, token]
    cx3 = sqlite3.connect(db); cx3.row_factory = sqlite3.Row
    row3 = O.find_order_by_external_ref(cx3, token)
    assert row3["qbo_sales_receipt_id"] == "SR1"


def test_invoice_based_reorder_return_still_records_qbo_payment(monkeypatch, tmp_path):
    """PINNING test: a NOT-yet-converted invoice-based order (kind=="reorder" but
    with a REAL non-empty QBO customer_id in the Stripe metadata -- i.e. a legacy
    or not-yet-migrated portal-reorder/reorder checkout that still create_invoice'd
    against a real QBO customer) must still get record_payment applied on return.

    Commit 8d7c33d0 wrongly added "reorder"/"portal-reorder"/"subscribe" to the
    record_payment exclusion kind-list, which would skip record_payment here even
    though `cid` is real and there's an actual QBO invoice to apply the payment to.
    The exclusion must cover only ("biofield", "retail") -- the leading `if cid`
    already makes the guard a no-op for the converted paid-only kinds (their
    metadata customer_id is ""), so kind-based exclusion beyond biofield/retail is
    both wrong and unnecessary."""
    db = _isolate_db(monkeypatch, tmp_path)
    token = "chk_pin_invoice_reorder_1"
    real_cid = "C-REAL-999"
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    try:
        O.upsert_order(cx, source="reorder", external_ref=token, email="legacy@b.com",
                       name="Legacy", items=[{"slug": PRODUCT_SLUG, "qty": 1}],
                       total_cents=7000, address={}, channel="retail",
                       get_cents=0, discount_cents=0, points_redeemed_cents=0,
                       shipping_cents=0, status="new")
    finally:
        cx.close()

    monkeypatch.setattr(app, "_settle_order_points", lambda order, *, order_ref: None)
    monkeypatch.setattr(app, "_settle_referral", lambda order, *, order_ref: None)

    calls = {"record_payment": []}

    def spy_record_payment(cid, amount_cents, inv):
        calls["record_payment"].append((cid, amount_cents, inv))
    monkeypatch.setattr(qbo_billing, "record_payment", spy_record_payment)

    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "get_session", lambda sid: {
        "payment_status": "paid", "amount_total": 7000, "payment_intent": "pi_legacy",
        "metadata": {"kind": "reorder", "invoice_id": token, "customer_id": real_cid}})

    r = _client().get("/begin/checkout-return?session_id=sess1")
    assert r.status_code in (301, 302)

    assert calls["record_payment"] == [(real_cid, 7000, token)]
