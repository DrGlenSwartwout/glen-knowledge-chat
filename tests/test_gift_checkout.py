# tests/test_gift_checkout.py
import sqlite3
import pytest


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod, "LOG_DB", str(tmp_path / "chat_log.db"))
    monkeypatch.setattr(appmod, "REWARDS_1B_GIFT_ENABLED", True)
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        coupons.init_coupons_table(cx)
    return appmod


def test_resolve_gift_pct_and_self_block(appmod):
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        g = coupons.mint_gift(cx, email="owner@x.com", product_slug="terrain-restore")
    pct, found = appmod._resolve_gift_coupon_pct(g["code"], "friend@y.com")
    assert pct == 15 and found and found["code"] == g["code"]
    assert found["email"] == "owner@x.com"
    # owner cannot redeem their own gift
    assert appmod._resolve_gift_coupon_pct(g["code"], "owner@x.com")[0] == 0
    assert appmod._resolve_gift_coupon_pct("nope", "friend@y.com")[0] == 0


def test_self_autoapply_ignores_gift_coupons(appmod):
    """A gift coupon owned by the same email must NOT be auto-applied as a self discount."""
    from dashboard import coupons
    with sqlite3.connect(appmod.LOG_DB) as cx:
        coupons.mint_gift(cx, email="me@x.com", product_slug="terrain-restore")  # only a gift, no self
    assert appmod._best_active_self_coupon_code("me@x.com", "terrain-restore") == ""
