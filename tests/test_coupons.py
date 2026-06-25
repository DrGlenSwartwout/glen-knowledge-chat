import sqlite3
import pytest
from dashboard import coupons


@pytest.fixture
def cx(tmp_path):
    c = sqlite3.connect(str(tmp_path / "t.db"))
    coupons.init_coupons_table(c)
    return c


def test_mint_self_is_idempotent_while_active(cx):
    a = coupons.mint_self(cx, email="m@x.com", product_slug="terrain-restore")
    b = coupons.mint_self(cx, email="m@x.com", product_slug="terrain-restore")
    assert a["code"] == b["code"]
    assert a["pct"] == 15 and a["kind"] == "self"


def test_validate_ok_and_product_mismatch(cx):
    c = coupons.mint_self(cx, email="m@x.com", product_slug="terrain-restore")
    assert coupons.validate(cx, c["code"], product_slug="terrain-restore")
    assert coupons.validate(cx, c["code"], product_slug="other-slug") is None


def test_validate_expired_is_none(cx):
    c = coupons.mint_self(cx, email="m@x.com", product_slug="x", days=-1)  # already expired
    assert coupons.validate(cx, c["code"]) is None


def test_mark_redeemed_then_invalid(cx):
    c = coupons.mint_self(cx, email="m@x.com", product_slug="x")
    assert coupons.mark_redeemed(cx, c["code"], order_ref="INV-1") is True
    assert coupons.mark_redeemed(cx, c["code"], order_ref="INV-1") is False  # idempotent
    assert coupons.validate(cx, c["code"]) is None


def test_wallet_lists_active_only(cx):
    coupons.mint_self(cx, email="m@x.com", product_slug="a")
    dead = coupons.mint_self(cx, email="m@x.com", product_slug="b", days=-1)
    w = coupons.wallet(cx, email="m@x.com")
    slugs = {r["product_slug"] for r in w}
    assert slugs == {"a"} and dead["product_slug"] == "b"


def test_mint_self_blocks_remint_after_redeem(cx):
    a = coupons.mint_self(cx, email="m@x.com", product_slug="x")
    coupons.mark_redeemed(cx, a["code"], order_ref="INV-1")
    b = coupons.mint_self(cx, email="m@x.com", product_slug="x")
    assert b["code"] == a["code"]  # earn-once: no fresh coupon after redemption


def test_mint_self_remints_after_unused_expiry(cx):
    a = coupons.mint_self(cx, email="m@x.com", product_slug="x", days=-1)  # expired, unused
    b = coupons.mint_self(cx, email="m@x.com", product_slug="x")
    assert b["code"] != a["code"]  # a second chance if the first expired unused
