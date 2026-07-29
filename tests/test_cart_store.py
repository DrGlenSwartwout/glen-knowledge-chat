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


def test_release_or_fold_never_folds_a_cart_into_itself(cx):
    """`release_or_fold_stale_claim`'s `token<>?` exclusion is not merely defensive.
    Under a concurrent stale-recovery race the cart can already be back to 'open' by
    the time this query runs; without the exclusion the "member's OTHER open cart"
    lookup returns THIS cart, and `_fold_cart_items(cx, tok, tok)` folds it into
    itself -- deleting every item and marking the cart 'merged'. Total item loss on
    the money path."""
    CS.get_or_create(cx, "tokA", email="a@x.com")
    CS.add_item(cx, "tokA", "brain-boost", qty=2)
    CS.add_item(cx, "tokA", "wholomega", qty=1)

    CS.release_or_fold_stale_claim(cx, "tokA", "a@x.com")

    assert {i["slug"]: i["qty"] for i in CS.items(cx, "tokA")} == {
        "brain-boost": 2, "wholomega": 1}
    assert cx.execute(
        "SELECT status FROM carts WHERE token=?", ("tokA",)).fetchone()[0] == "open"


def test_release_claim_will_not_reopen_an_already_ordered_cart(cx):
    """`release_claim`'s `AND status='checking_out'` clause is what stops an
    already-'ordered' cart being reopened. A cart whose order exists and whose
    customer already has a payment link must never go back to 'open' -- that is a
    second charge for the same items. Every release path (bad address, CheckoutError,
    the catch-all handler, stale-claim recovery) can fire against a cart that has
    since been closed, so the clause is the guard, not the call sites."""
    CS.get_or_create(cx, "tok1", email="a@x.com")
    CS.add_item(cx, "tok1", "brain-boost", qty=2)
    assert CS.claim_for_checkout(cx, "tok1") is True
    CS.mark_ordered(cx, "tok1", "INV-1")

    CS.release_claim(cx, "tok1")

    status, ref = cx.execute(
        "SELECT status, checkout_ref FROM carts WHERE token=?", ("tok1",)).fetchone()
    assert status == "ordered"
    assert ref == "INV-1"
