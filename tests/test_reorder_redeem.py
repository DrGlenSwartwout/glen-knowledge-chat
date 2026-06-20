import sqlite3
import app as appmod
import begin_funnel
from dashboard import points


def _setup(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db"); monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db); points.init_points_table(cx)
    begin_funnel.init_journey_tables(cx)
    begin_funnel.record_unlock(cx, session_id="sess-redeem-test", trigger="tos",
                               email="a@x.com", tos=True)
    points.earn(cx, "a@x.com", full_price_cents=40000, earn_pct=0.05, order_ref="s"); cx.commit()  # 2000
    monkeypatch.setattr(appmod, "_reorder_email_from_cookie", lambda: "a@x.com")
    monkeypatch.setattr(appmod, "_get_product",
        lambda s: {"slug": s, "name": "Brain Boost", "price_cents": 7000, "qty_pricing": True, "qbo_item_id": "27"} if s == "brain-boost" else None)
    monkeypatch.setattr(appmod._shipping, "quote", lambda b: {"shipping_cents": 0})
    monkeypatch.setattr(appmod.qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    cap = {}
    monkeypatch.setattr(appmod.qb, "create_invoice",
        lambda *a, **k: cap.update(k) or {"Id": "INV", "TotalAmt": 68.0})
    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: cap.setdefault("order", kw))
    monkeypatch.setattr(appmod, "_stripe_checkout_url_for_reorder", lambda *a, **k: "")
    monkeypatch.setenv("PRICING_ENGINE_CHECKOUT", "true")
    return cap


def test_reorder_redeems_points(monkeypatch, tmp_path):
    cap = _setup(monkeypatch, tmp_path)
    c = appmod.app.test_client()
    r = c.post("/reorder/checkout", json={"items": [{"slug": "brain-boost", "qty": 1}],
               "address": {"state": "CA", "country": "US", "name": "A"}, "points_to_redeem_cents": 200})
    assert r.status_code == 200
    # 1 bottle $70 full price, 200 pts -> line 6800; order records 200 redeemed
    assert cap["order"]["points_redeemed_cents"] == 200


def test_reorder_caps_redemption_to_balance(monkeypatch, tmp_path):
    cap = _setup(monkeypatch, tmp_path)   # balance 2000
    c = appmod.app.test_client()
    r = c.post("/reorder/checkout", json={"items": [{"slug": "brain-boost", "qty": 1}],
               "address": {"state": "CA", "country": "US", "name": "A"}, "points_to_redeem_cents": 999999})
    assert r.status_code == 200
    assert cap["order"]["points_redeemed_cents"] <= 2000   # never more than they have
