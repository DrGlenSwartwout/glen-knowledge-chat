# tests/test_subscribe_setup.py
import app as appmod

def _setup(monkeypatch):
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "a@x.com")
    monkeypatch.setattr(appmod, "_get_product",
        lambda s: {"slug":s,"name":"Brain Boost","price_cents":7000,"qty_pricing":True,"qbo_item_id":"27"} if s=="brain-boost" else None)
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 2295})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(appmod.qb, "create_invoice", lambda *a, **k: {"Id":"INV1","TotalAmt":50.0,"DocNumber":"1"})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)
    cap = {}
    monkeypatch.setattr(appmod.stripe_pay, "create_checkout_session",
        lambda *a, **k: cap.update(k) or {"id":"cs_1","url":"https://stripe/setup"})
    monkeypatch.setenv("PRICING_ENGINE_CHECKOUT", "true")
    monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "true")
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True)
    return cap

def test_subscribe_creates_setup_session_with_save_card(monkeypatch):
    cap = _setup(monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/reorder/subscribe", json={"items":[{"slug":"brain-boost","qty":1}],
               "cadence_months":1, "address":{"state":"CA","country":"US","name":"A"}})
    assert r.status_code == 200
    assert r.get_json()["stripe_url"] == "https://stripe/setup"
    assert cap["save_card"] is True                      # vaults the card
    # first order priced at the 5% tier (tier_for(0))
    # (invoice stub fixed; assert the metadata carries the kind + cadence for the return handler)
    assert cap["metadata"]["kind"] == "subscribe"
    assert cap["metadata"]["cadence_months"] == "1"

def test_subscribe_disabled_when_flag_off(monkeypatch):
    cap = _setup(monkeypatch); monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "false")
    c = appmod.app.test_client()
    r = c.post("/reorder/subscribe", json={"items":[{"slug":"brain-boost","qty":1}],
               "cadence_months":1, "address":{"state":"CA","country":"US"}})
    assert r.status_code == 400
