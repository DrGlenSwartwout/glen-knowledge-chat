# tests/test_reorder_checkout_engine.py
import app as appmod

def _setup(monkeypatch):
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "a@x.com")
    monkeypatch.setattr(appmod, "_get_product",
        lambda s: {"slug":s,"name":"Brain Boost","price_cents":7000,"qty_pricing":True,"qbo_item_id":"27"} if s=="brain-boost" else None)
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 2295})
    captured = {}
    def fake_invoice(cust, lines, **kw):
        captured["lines"] = lines; captured["kw"] = kw
        return {"Id": "INV9", "TotalAmt": 100.0}
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice", fake_invoice)
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: captured.setdefault("order", kw))
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_reorder", lambda *a, **k: "https://stripe/x")
    monkeypatch.setenv("PRICING_ENGINE_CHECKOUT", "true")
    return captured

def test_reorder_checkout_uses_engine_discount(monkeypatch):
    captured = _setup(monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/reorder/checkout", json={"items":[{"slug":"brain-boost","qty":6}],
                                          "address":{"state":"CA","country":"US","name":"A"}})
    assert r.status_code == 200
    # engine discount (linear ~19.545% off 42000 -> 42000-33791=8209) passed to QBO
    assert captured["kw"]["discount_cents"] == 8209
    assert captured["order"]["discount_cents"] == 8209
    assert captured["order"]["shipping_cents"] == 2295
    # customer_id must be echoed so /begin/checkout-return can record the QBO payment
    assert r.get_json()["customer_id"] == "C1"


def test_reorder_checkout_card_failure_surfaces_payment_error(monkeypatch):
    _setup(monkeypatch)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    # Stripe failed -> helper swallowed it + returned "" (the _alert_stripe path).
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_reorder", lambda *a, **k: "")
    c = appmod.app.test_client()
    r = c.post("/reorder/checkout", json={"items": [{"slug": "brain-boost", "qty": 6}],
                                          "address": {"state": "CA", "country": "US", "name": "A"}})
    assert r.status_code == 200            # graceful, not a 500
    body = r.get_json()
    assert body["ok"] is True
    assert body["stripe_url"] == ""
    assert body["payment_error"] == appmod._CARD_UNAVAILABLE


def test_reorder_checkout_success_has_no_payment_error(monkeypatch):
    _setup(monkeypatch)  # mocks the helper -> a real URL
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    c = appmod.app.test_client()
    r = c.post("/reorder/checkout", json={"items": [{"slug": "brain-boost", "qty": 6}],
                                          "address": {"state": "CA", "country": "US", "name": "A"}})
    assert r.status_code == 200
    body = r.get_json()
    assert body["stripe_url"] == "https://stripe/x"
    assert "payment_error" not in body     # success shape unchanged
