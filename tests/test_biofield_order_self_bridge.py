import sqlite3
import app as app_module
from dashboard import recommendation_events as re


def test_reveal_order_emits_biofield_events(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    cx = sqlite3.connect(db); re.init_recommendation_events(cx); cx.close()
    monkeypatch.setattr(app_module, "LOG_DB", db, raising=False)
    monkeypatch.setattr(app_module, "BIOFIELD_CART_ENABLED", True, raising=False)
    monkeypatch.setattr(app_module, "_biofield_verify_token", lambda th: (True, {"id": 1, "email": "a@b.com"}))
    monkeypatch.setattr(app_module, "_biofield_visible_slugs", lambda row, email: {"neuro-magnesium", "immune-modulation"})
    monkeypatch.setattr(app_module, "is_member", lambda *a, **k: True)
    monkeypatch.setattr(app_module, "_resolve_ship_address", lambda *a, **k: {})
    monkeypatch.setattr(app_module, "_checkout_cart", lambda email, items, **k: {"out": {}, "stripe_url": "https://x"})
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    r = c.post("/begin/biofield/anytok/order-checkout",
               json={"items": [{"slug": "neuro-magnesium", "qty": 1}, {"slug": "immune-modulation", "qty": 2}]})
    assert r.get_json()["ok"] is True
    cx = sqlite3.connect(db)
    bf = {e["product_key"] for e in re.list_events(cx, "a@b.com") if e["source_key"] == "biofield"}
    assert bf == {"neuro-magnesium", "immune-modulation"}
