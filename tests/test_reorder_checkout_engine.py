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
    # engine discount (6*(7000-4970)=12180) passed to QBO as discount_cents
    assert captured["kw"]["discount_cents"] == 12180
    assert captured["order"]["discount_cents"] == 12180
    assert captured["order"]["shipping_cents"] == 2295
