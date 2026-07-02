# tests/test_begin_checkout_engine.py
import app as appmod

def _setup(monkeypatch):
    monkeypatch.setattr(appmod, "is_member", lambda sid, email: True)   # consent satisfied
    monkeypatch.setattr(appmod, "_get_product",
        lambda s: {"slug":s,"name":"Brain Boost","price_cents":7000,"qty_pricing":True,"qbo_item_id":"27"} if s=="brain-boost" else None)
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 2295})
    monkeypatch.setattr(appmod.qb if hasattr(appmod,"qb") else appmod, "find_or_create_customer", lambda *a, **k: {"Id":"C1"}, raising=False)
    cap = {}
    def fake_invoice(cust, lines, **kw):
        cap["lines"] = lines; cap["kw"] = kw
        return {"Id":"INV","TotalAmt":74.0,"DocNumber":"7"}
    # qbo_billing is imported locally in begin_checkout; patch the module it imports
    import dashboard.qbo_billing as _qb
    monkeypatch.setattr(_qb, "find_or_create_customer", lambda *a, **k: {"Id":"C1"})
    monkeypatch.setattr(_qb, "create_invoice", fake_invoice)
    monkeypatch.setattr(_qb, "get_invoice_pay_link", lambda inv: "")
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: cap.setdefault("order", kw))
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_retail", lambda *a, **k: "https://stripe/x")
    monkeypatch.setenv("PRICING_ENGINE_CHECKOUT", "true")
    return cap

def test_begin_checkout_engine_records_discount_and_shipping(monkeypatch):
    cap = _setup(monkeypatch)
    c = appmod.app.test_client()
    r = c.post("/begin/checkout/brain-boost", json={
        "email":"buyer@x.com","name":"B","method":"card","qty":6,
        "address":{"state":"CA","country":"US","name":"B"}})
    assert r.status_code == 200
    # 6 units → volume ~24.333% off 42000 → discount 42000-31780=10220 passed to QBO
    assert cap["kw"]["discount_cents"] == 10220
    assert cap["order"]["discount_cents"] == 10220
    assert cap["order"]["shipping_cents"] == 2295
    assert cap["order"]["source"] == "funnel"
    assert r.get_json()["customer_id"] == "C1"

def test_begin_checkout_consent_gate_still_enforced(monkeypatch):
    cap = _setup(monkeypatch)
    monkeypatch.setattr(appmod, "is_member", lambda sid, email: False)
    c = appmod.app.test_client()
    r = c.post("/begin/checkout/brain-boost", json={"email":"b@x.com","method":"card",
               "address":{"state":"CA","country":"US"}})
    assert r.status_code == 403 and r.get_json().get("need_optin") is True
