# tests/test_price_cart.py
import pytest, app as appmod

def _stub_products(monkeypatch):
    cat = {"brain-boost": {"slug":"brain-boost","name":"Brain Boost","price_cents":7000,
                           "qty_pricing":True,"qbo_item_id":"27"}}
    monkeypatch.setattr(appmod, "_get_product", lambda s: cat.get(s))

def test_price_cart_volume_and_shipping(monkeypatch):
    _stub_products(monkeypatch)
    # 6 units total → LINEAR volume 13.1818% off the 42000 line
    # line_total_cents = round(42000*(1-0.131818)) = 36464; discount = 42000 - 36464 = 5536
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 2295, "box": "M"})
    out = appmod._price_cart([{"slug":"brain-boost","qty":6}], ship={"state":"CA","country":"US"})
    assert out["priced"]["lines"][0]["line_total_cents"] == 36464
    assert out["discount_cents"] == 5536                      # engine discount, list - net
    assert out["shipping_cents"] == 2295
    # QBO lines carry LIST price (qty applied by QBO), discount is separate
    assert out["qbo_lines"][0]["amount"] == 70.0 and out["qbo_lines"][0]["qty"] == 6

def test_price_cart_rejects_non_us(monkeypatch):
    _stub_products(monkeypatch)
    with pytest.raises(appmod.CheckoutError):
        appmod._price_cart([{"slug":"brain-boost","qty":1}], ship={"state":"ON","country":"CA"})

def test_price_cart_skips_unknown(monkeypatch):
    _stub_products(monkeypatch)
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 0})
    out = appmod._price_cart([{"slug":"nope","qty":1}], ship={"state":"CA","country":"US"})
    assert out["qbo_lines"] == []
