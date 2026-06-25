"""Self-coupon resolution + that the existing pricing path applies it under the floor."""
import sqlite3
import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "REWARDS_1B_ENABLED", True)
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        coupons.init_coupons_table(cx)
    return appmod


def test_resolve_self_coupon_pct(appmod):
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        c = coupons.mint_self(cx, email="m@x.com", product_slug="terrain-restore")
    pct, found = appmod._resolve_self_coupon_pct(c["code"], "terrain-restore")
    assert pct == 15 and found and found["code"] == c["code"]
    # wrong product → no match
    assert appmod._resolve_self_coupon_pct(c["code"], "other")[0] == 0
    # junk code → no match
    assert appmod._resolve_self_coupon_pct("nope", "terrain-restore")[0] == 0


def test_pricing_applies_coupon_clamped_to_floor(appmod):
    from dashboard import pricing
    # 90% off would blow past the 57% wholesale floor → clamp up to the floor
    p = {"name": "X", "price_cents": 10000, "sku_discount_floor_pct": 0.57}
    # item shape must match what _engine_item / pricing.compute expects:
    # unit_cents (not unit_list_cents), plus slug and name at the item level
    item = {"product": p, "qty": 1, "unit_cents": 10000, "slug": "x", "name": "X"}
    out = pricing.compute([item], settings=pricing.DEFAULTS, coupon_pct=90)
    # compute returns line_total_cents (not unit_cents); for qty=1 this is the effective unit price
    assert out["lines"][0]["line_total_cents"] >= 5700  # never below the 57% floor
