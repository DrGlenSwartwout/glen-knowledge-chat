# tests/test_subscribe_setup.py
import app as appmod

def _setup(monkeypatch):
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "a@x.com")
    monkeypatch.setattr(appmod, "_get_product",
        lambda s: {"slug":s,"name":"Brain Boost","price_cents":7000,"qty_pricing":True,"qbo_item_id":"27"} if s=="brain-boost" else None)
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 2295})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    # Paid-only (QBO Stage 3): reorder_subscribe no longer calls create_invoice --
    # guard it (mutation-style), and stub set_order_qbo_lines so the persisted
    # payload doesn't touch the real local chat_log.db in this un-isolated fixture.
    def boom(*a, **k):
        raise AssertionError("reorder_subscribe must not call create_invoice (paid-only)")
    monkeypatch.setattr(appmod.qb, "create_invoice", boom)
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: None)
    monkeypatch.setattr(appmod._bos_orders, "set_order_qbo_lines", lambda cx, ref, payload: True)
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
    # paid-only: no QBO invoice/customer at checkout time -- metadata carries a
    # token invoice_id (checkout_ref) and empty customer_id, plus kind + cadence
    # for the return handler.
    assert cap["metadata"]["kind"] == "subscribe"
    assert cap["metadata"]["cadence_months"] == "1"
    assert cap["metadata"]["invoice_id"]
    assert cap["metadata"]["customer_id"] == ""

def test_subscribe_disabled_when_flag_off(monkeypatch):
    cap = _setup(monkeypatch); monkeypatch.setenv("SUBSCRIPTIONS_ENABLED", "false")
    c = appmod.app.test_client()
    r = c.post("/reorder/subscribe", json={"items":[{"slug":"brain-boost","qty":1}],
               "cadence_months":1, "address":{"state":"CA","country":"US"}})
    assert r.status_code == 400

def test_subscribe_rejected_when_stripe_inactive(monkeypatch):
    # must 400 BEFORE creating any QBO invoice (no dangling invoice) -- and,
    # paid-only, before any order is even ingested.
    cap = _setup(monkeypatch); monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", False)
    inv_called = {"n": 0}
    monkeypatch.setattr(appmod.qb, "create_invoice",
                        lambda *a, **k: inv_called.update(n=inv_called["n"] + 1) or {"Id": "X"})
    c = appmod.app.test_client()
    r = c.post("/reorder/subscribe", json={"items":[{"slug":"brain-boost","qty":1}],
               "cadence_months":1, "address":{"state":"CA","country":"US"}})
    assert r.status_code == 400
    assert inv_called["n"] == 0          # invoice never created when Stripe is off
