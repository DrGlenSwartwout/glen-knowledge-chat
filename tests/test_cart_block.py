import sqlite3

import pytest

from dashboard import cart_block as CB
from dashboard import cart_store as CS


@pytest.fixture()
def cx(tmp_path):
    c = sqlite3.connect(str(tmp_path / "cart.db"))
    CS.init_cart_tables(c)
    yield c
    c.close()


def test_block_is_inert_when_disabled(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=3)
    assert CB.build_block(cx, "a@x.com", False) == {"enabled": False}


def test_block_counts_the_open_cart(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=3)
    CS.add_item(cx, "mem1", "wholomega", qty=2)
    assert CB.build_block(cx, "a@x.com", True) == {"enabled": True, "count": 5}


def test_block_is_zero_when_the_member_has_no_cart(cx):
    assert CB.build_block(cx, "nobody@x.com", True) == {"enabled": True, "count": 0}


def test_block_ignores_an_ordered_cart(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=3)
    CS.mark_ordered(cx, "mem1", "ref1")
    assert CB.build_block(cx, "a@x.com", True) == {"enabled": True, "count": 0}


def test_block_never_raises_into_the_payload(cx):
    """A portal payload must degrade, not 500, when a source fails.

    Creates a real open cart with items FIRST, so open_token_for_email
    resolves a non-empty token and the subsequent items() lookup actually
    hits the dropped table -- otherwise this would pass for the wrong
    reason (an email with no cart at all short-circuits to 0 without ever
    touching cart_items, never exercising the guard this test claims to
    cover)."""
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=3)
    cx.execute("DROP TABLE cart_items")
    cx.commit()
    assert CB.build_block(cx, "a@x.com", True) == {"enabled": True, "count": 0}


def test_hub_tile_is_gated_on_enabled():
    html = open("static/client-portal.html", encoding="utf-8").read()
    assert "v.cart && v.cart.enabled" in html
