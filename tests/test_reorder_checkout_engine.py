# tests/test_reorder_checkout_engine.py
import sqlite3
import app as appmod
import begin_funnel

def _setup(monkeypatch):
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "a@x.com")
    with sqlite3.connect(appmod.LOG_DB) as _cx:
        begin_funnel.init_journey_tables(_cx)
        begin_funnel.record_unlock(_cx, session_id="sess-reorder-engine-test", trigger="tos",
                                   email="a@x.com", tos=True)
    monkeypatch.setattr(appmod, "_get_product",
        lambda s: {"slug":s,"name":"Brain Boost","price_cents":7000,"qty_pricing":True,"qbo_item_id":"27"} if s=="brain-boost" else None)
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 2295})
    captured = {}
    def boom(*a, **k):
        raise AssertionError("_checkout_cart must not call create_invoice (paid-only)")
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice", boom)
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: captured.setdefault("order", kw))
    # _checkout_cart is paid-only (QBO Stage 2): it no longer calls create_invoice, so
    # the QBO line/discount payload is persisted via set_order_qbo_lines instead --
    # capture it the same way `order` is captured above.
    monkeypatch.setattr(appmod._bos_orders, "set_order_qbo_lines",
                        lambda cx, ref, payload: captured.setdefault("qbo_payload", payload))
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_reorder", lambda *a, **k: "https://stripe/x")
    monkeypatch.setenv("PRICING_ENGINE_CHECKOUT", "true")
    return captured

def test_reorder_checkout_uses_engine_discount(monkeypatch):
    captured = _setup(monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/reorder/checkout", json={"items":[{"slug":"brain-boost","qty":6}],
                                          "address":{"state":"CA","country":"US","name":"A"}})
    assert r.status_code == 200
    # engine discount (LINEAR 13.1818% off 42000 -> 42000-36464=5536) passed to QBO
    assert captured["qbo_payload"]["discount_cents"] == 5536
    assert captured["order"]["discount_cents"] == 5536
    assert captured["order"]["shipping_cents"] == 2295
    # paid-only: no real QBO customer exists at checkout time
    assert r.get_json()["customer_id"] == ""


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
