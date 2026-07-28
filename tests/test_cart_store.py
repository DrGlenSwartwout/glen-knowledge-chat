import sqlite3

import pytest

from dashboard import cart_store as CS


@pytest.fixture()
def cx(tmp_path):
    c = sqlite3.connect(str(tmp_path / "cart.db"))
    CS.init_cart_tables(c)
    yield c
    c.close()


def test_get_or_create_returns_token_and_is_idempotent(cx):
    t = CS.get_or_create(cx, "tok1")
    assert t == "tok1"
    assert CS.get_or_create(cx, "tok1") == "tok1"
    assert CS.items(cx, "tok1") == []


def test_add_item_then_list(cx):
    CS.get_or_create(cx, "tok1")
    assert CS.add_item(cx, "tok1", "brain-boost", qty=2, source="product") == 2
    assert CS.items(cx, "tok1") == [
        {"slug": "brain-boost", "qty": 2, "format": "", "source": "product"}
    ]


def test_add_same_slug_and_format_increments(cx):
    CS.get_or_create(cx, "tok1")
    CS.add_item(cx, "tok1", "brain-boost", qty=1)
    assert CS.add_item(cx, "tok1", "brain-boost", qty=2) == 3
    assert len(CS.items(cx, "tok1")) == 1


def test_same_slug_different_format_is_a_separate_line(cx):
    CS.get_or_create(cx, "tok1")
    CS.add_item(cx, "tok1", "brain-boost", qty=1, fmt="bottle")
    CS.add_item(cx, "tok1", "brain-boost", qty=1, fmt="refill")
    assert len(CS.items(cx, "tok1")) == 2


def test_set_qty_updates_and_zero_removes(cx):
    CS.get_or_create(cx, "tok1")
    CS.add_item(cx, "tok1", "brain-boost", qty=5)
    CS.set_qty(cx, "tok1", "brain-boost", "", 2)
    assert CS.items(cx, "tok1")[0]["qty"] == 2
    CS.set_qty(cx, "tok1", "brain-boost", "", 0)
    assert CS.items(cx, "tok1") == []


def test_qty_is_clamped_to_1_99(cx):
    CS.get_or_create(cx, "tok1")
    CS.add_item(cx, "tok1", "brain-boost", qty=500)
    assert CS.items(cx, "tok1")[0]["qty"] == 99


def test_open_token_for_email(cx):
    CS.get_or_create(cx, "tokA", email="A@X.com")
    assert CS.open_token_for_email(cx, "a@x.com") == "tokA"
    assert CS.open_token_for_email(cx, "nobody@x.com") == ""


def test_get_or_create_returns_the_members_existing_open_cart(cx):
    CS.get_or_create(cx, "tokA", email="a@x.com")
    result = CS.get_or_create(cx, "tokB", email="a@x.com")
    assert result == "tokA"
    assert CS.items(cx, "tokA") == []


def test_get_or_create_after_that_cart_was_ordered_mints_a_new_token(cx):
    tok1 = CS.get_or_create(cx, "tok1")
    assert tok1 == "tok1"
    cx.execute("UPDATE carts SET status='ordered' WHERE token=?", ("tok1",))
    cx.commit()
    new_tok = CS.get_or_create(cx, "tok1")
    assert new_tok
    assert new_tok != "tok1"
    assert CS.items(cx, new_tok) == []


def test_get_or_create_is_still_idempotent_for_an_open_token(cx):
    t = CS.get_or_create(cx, "tok1")
    assert t == "tok1"
    assert CS.get_or_create(cx, "tok1") == "tok1"
