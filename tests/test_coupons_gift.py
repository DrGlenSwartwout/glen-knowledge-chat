import sqlite3
import pytest
from dashboard import coupons


@pytest.fixture
def cx(tmp_path):
    c = sqlite3.connect(str(tmp_path / "t.db"))
    coupons.init_coupons_table(c)
    return c


def test_mint_gift_idempotent_and_prefixed(cx):
    a = coupons.mint_gift(cx, email="owner@x.com", product_slug="terrain-restore")
    b = coupons.mint_gift(cx, email="owner@x.com", product_slug="terrain-restore")
    assert a["code"] == b["code"]
    assert a["kind"] == "gift" and a["code"].startswith("GIFT-") and a["pct"] == 15


def test_validate_gift_ok_for_other_email(cx):
    c = coupons.mint_gift(cx, email="owner@x.com", product_slug="terrain-restore")
    assert coupons.validate_gift(cx, c["code"], referee_email="friend@y.com")


def test_validate_gift_blocks_self_gift(cx):
    c = coupons.mint_gift(cx, email="owner@x.com", product_slug="terrain-restore")
    assert coupons.validate_gift(cx, c["code"], referee_email="OWNER@x.com") is None  # case-insensitive


def test_validate_gift_rejects_self_kind_code(cx):
    s = coupons.mint_self(cx, email="owner@x.com", product_slug="terrain-restore")
    assert coupons.validate_gift(cx, s["code"], referee_email="friend@y.com") is None


def test_validate_gift_expired_and_redeemed(cx):
    dead = coupons.mint_gift(cx, email="o@x.com", product_slug="p", days=-1)
    assert coupons.validate_gift(cx, dead["code"], referee_email="f@y.com") is None
    live = coupons.mint_gift(cx, email="o@x.com", product_slug="q")
    coupons.mark_redeemed(cx, live["code"], order_ref="INV-1")
    assert coupons.validate_gift(cx, live["code"], referee_email="f@y.com") is None


def test_wallet_kind_filter(cx):
    coupons.mint_self(cx, email="o@x.com", product_slug="a")
    coupons.mint_gift(cx, email="o@x.com", product_slug="a")
    assert {c["kind"] for c in coupons.wallet(cx, email="o@x.com")} == {"self", "gift"}
    assert {c["kind"] for c in coupons.wallet(cx, email="o@x.com", kind="self")} == {"self"}
    assert {c["kind"] for c in coupons.wallet(cx, email="o@x.com", kind="gift")} == {"gift"}
