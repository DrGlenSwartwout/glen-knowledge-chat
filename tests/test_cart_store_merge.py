import sqlite3

import pytest

from dashboard import cart_store as CS


@pytest.fixture()
def cx(tmp_path):
    c = sqlite3.connect(str(tmp_path / "cart.db"))
    CS.init_cart_tables(c)
    yield c
    c.close()


def test_merge_claims_anon_cart_when_member_has_none(cx):
    CS.get_or_create(cx, "anon1")
    CS.add_item(cx, "anon1", "brain-boost", qty=2)
    surviving = CS.merge(cx, "anon1", "A@X.com")
    assert surviving == "anon1"
    assert CS.open_token_for_email(cx, "a@x.com") == "anon1"
    assert CS.items(cx, "anon1")[0]["qty"] == 2


def test_merge_folds_into_existing_member_cart_higher_qty_wins(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=1)
    CS.add_item(cx, "mem1", "wholomega", qty=4)
    CS.get_or_create(cx, "anon1")
    CS.add_item(cx, "anon1", "brain-boost", qty=3)   # higher than member's 1
    CS.add_item(cx, "anon1", "neuroprotect", qty=1)  # new line

    surviving = CS.merge(cx, "anon1", "a@x.com")

    assert surviving == "mem1"
    got = {i["slug"]: i["qty"] for i in CS.items(cx, "mem1")}
    assert got == {"brain-boost": 3, "wholomega": 4, "neuroprotect": 1}


def test_merge_never_sums(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=2)
    CS.get_or_create(cx, "anon1")
    CS.add_item(cx, "anon1", "brain-boost", qty=2)
    CS.merge(cx, "anon1", "a@x.com")
    assert CS.items(cx, "mem1")[0]["qty"] == 2


def test_merge_closes_the_anon_cart(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.get_or_create(cx, "anon1")
    CS.add_item(cx, "anon1", "brain-boost", qty=1)
    CS.merge(cx, "anon1", "a@x.com")
    assert CS.items(cx, "anon1") == []
    row = cx.execute("SELECT status FROM carts WHERE token=?", ("anon1",)).fetchone()
    assert row[0] == "merged"


def test_merge_is_idempotent(cx):
    CS.get_or_create(cx, "anon1")
    CS.add_item(cx, "anon1", "brain-boost", qty=2)
    first = CS.merge(cx, "anon1", "a@x.com")
    second = CS.merge(cx, "anon1", "a@x.com")
    assert first == second == "anon1"
    assert CS.items(cx, first)[0]["qty"] == 2


def test_merge_with_unknown_anon_token_returns_member_cart(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    assert CS.merge(cx, "nosuchtoken", "a@x.com") == "mem1"


def test_merge_requires_an_email(cx):
    CS.get_or_create(cx, "anon1")
    with pytest.raises(ValueError):
        CS.merge(cx, "anon1", "")


def test_mark_ordered_closes_cart_and_records_ref(cx):
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=1)
    CS.mark_ordered(cx, "mem1", "ref123")
    row = cx.execute(
        "SELECT status, checkout_ref FROM carts WHERE token=?", ("mem1",)
    ).fetchone()
    assert row[0] == "ordered"
    assert row[1] == "ref123"
    assert CS.open_token_for_email(cx, "a@x.com") == ""
