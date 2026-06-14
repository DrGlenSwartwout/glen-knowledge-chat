import sqlite3
from dashboard import points
import app as appmod


def _cx():
    cx = sqlite3.connect(":memory:"); cx.row_factory = sqlite3.Row
    points.init_points_table(cx); return cx


def test_has_entry_detects_prior_earn():
    cx = _cx()
    assert points.has_entry(cx, order_ref="INV1", reason="earn") is False
    points.earn(cx, "a@x.com", full_price_cents=7000, earn_pct=0.05, order_ref="INV1")
    assert points.has_entry(cx, order_ref="INV1", reason="earn") is True
    assert points.has_entry(cx, order_ref="INV1", reason="redeem") is False


def test_settle_points_earns_on_full_price(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    points.init_points_table(cx); cx.commit()
    order = {"email": "a@x.com", "total_cents": 7265, "shipping_cents": 1265,
             "discount_cents": 0, "points_redeemed_cents": 0}
    appmod._settle_order_points(order, order_ref="INV9")
    # product spend = 7265-1265 = 6000; earn 5% = 300
    assert points.balance(cx, "a@x.com") == 300
    # idempotent: second call does not double-earn
    appmod._settle_order_points(order, order_ref="INV9")
    assert points.balance(cx, "a@x.com") == 300


def test_settle_points_no_earn_when_discounted(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    points.init_points_table(cx); cx.commit()
    order = {"email": "a@x.com", "total_cents": 6000, "shipping_cents": 0,
             "discount_cents": 700, "points_redeemed_cents": 0}
    appmod._settle_order_points(order, order_ref="INV10")
    assert points.balance(cx, "a@x.com") == 0      # discounted -> full-price-only rule -> no earn


def test_settle_points_redeems_used_points(monkeypatch, tmp_path):
    db = str(tmp_path / "t.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    points.init_points_table(cx)
    points.earn(cx, "a@x.com", full_price_cents=20000, earn_pct=0.05, order_ref="seed")  # 1000
    cx.commit()
    order = {"email": "a@x.com", "total_cents": 5800, "shipping_cents": 0,
             "discount_cents": 0, "points_redeemed_cents": 200}
    appmod._settle_order_points(order, order_ref="INV11")
    # redeemed 200 deducted; NOT a full-price earn (points_redeemed>0 means a discount was applied)
    assert points.balance(cx, "a@x.com") == 800
