import re
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


def test_block_ignores_a_cart_mid_checkout(cx):
    """A cart claimed for checkout (status='checking_out') is deliberately NOT
    open (see cart_store.claim_for_checkout / open_token_for_email's status
    filter), so it must not show as a live cart -- a regression here would
    show a customer a live cart mid-payment."""
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=3)
    assert CS.claim_for_checkout(cx, "mem1") is True
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
    """Pinned to the ACTUAL actTiles.push(["cart" ...]) call site, not a bare
    substring search over the whole file. The gate condition
    `v.cart && v.cart.enabled` also appears at the separate panel-wrap call
    site (see test_cart_panel_is_gated_on_enabled below) -- a substring
    search that doesn't anchor to a specific call site can go green while
    ONE of the two occurrences is silently broken, because the other
    occurrence keeps the substring alive in the file. This regex requires
    the gate's `{` to be followed (allowing only whitespace) directly by
    `actTiles.push(["cart"`, so it can only be satisfied by the tile-push
    gate itself."""
    html = open("static/client-portal.html", encoding="utf-8").read()
    pattern = re.compile(
        r'if\s*\(\s*v\s*&&\s*v\.cart\s*&&\s*v\.cart\.enabled\s*\)\s*\{\s*'
        r'actTiles\.push\(\["cart"',
        re.DOTALL,
    )
    assert pattern.search(html), (
        "the actTiles.push for the cart tile must be immediately gated on "
        "v.cart.enabled"
    )


def test_cart_panel_is_gated_on_enabled():
    """The separate panel-wrap gate (` _hub && v.cart && v.cart.enabled ? `,
    guarding the [data-panel="cart"] <section>) is its own regression
    surface, distinct from the tile-push gate above. If this gate were
    dropped, the <section data-panel="cart"> would always render into the
    DOM even with the flag off. showTab() only bounces back to the hub when
    the target [data-panel] is ABSENT from the DOM -- so a stale
    sessionStorage('rm_portal_tab')=='cart' left over from a session where
    the flag was previously on (or any direct navigation to the cart tab)
    would land on a live (if now-empty/broken) cart panel instead of
    bouncing home, even though the flag is off and no tile is offered to
    reach it. That is a real, if narrow, off-state leak, so it gets its own
    pinned assertion rather than relying on the tile-push test to cover it."""
    html = open("static/client-portal.html", encoding="utf-8").read()
    pattern = re.compile(
        r'_hub\s*&&\s*v\.cart\s*&&\s*v\.cart\.enabled\s*\?\s*'
        r'`<section data-panel="cart"',
    )
    assert pattern.search(html), (
        "the cart panel section wrap must be gated on v.cart.enabled"
    )
