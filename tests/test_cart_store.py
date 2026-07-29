import sqlite3

import pytest

from dashboard import cart_store as CS
from tests.aborting_conn import AbortedTransaction, AbortingConn


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


def test_get_or_create_never_hands_back_another_members_open_cart(cx):
    """IMPORTANT (money): step 1 returned any open cart matching the token, with no
    ownership test. The token arrives from a browser cookie and this codebase models
    shared household devices, so the next visitor on that browser -- anonymous, or
    signed in as someone else -- was handed the previous member's cart and wrote
    into it. `merge` has carried this guard since an earlier round; step 1 did not."""
    CS.get_or_create(cx, "alice_cart", email="alice@x.com")
    CS.add_item(cx, "alice_cart", "brain-boost", qty=2)

    # Anonymous visitor presenting Alice's cookie token
    anon_tok = CS.get_or_create(cx, "alice_cart", email="")
    assert anon_tok != "alice_cart"
    assert CS.items(cx, anon_tok) == []

    # Bob, identified, presenting Alice's cookie token
    bob_tok = CS.get_or_create(cx, "alice_cart", email="bob@x.com")
    assert bob_tok not in ("alice_cart", anon_tok)
    assert CS.items(cx, bob_tok) == []

    # Alice's own cart is untouched, and she still gets it back
    assert {i["slug"]: i["qty"] for i in CS.items(cx, "alice_cart")} == {"brain-boost": 2}
    assert CS.get_or_create(cx, "alice_cart", email="Alice@X.com") == "alice_cart"


def test_get_or_create_race_recovery_survives_an_aborted_transaction(cx):
    """CRITICAL: `get_or_create`'s race self-heal is INERT on Postgres without the
    rollback. The `except Exception` recovers by re-reading, but psycopg leaves the
    transaction ABORTED after the failed INSERT, so that recovery SELECT -- the very
    next statement -- raises instead of returning the winner's token. The whole
    self-heal only ever worked because sqlite3 leaves the connection usable, and
    this repo has no Postgres lane to notice.

    `AbortingConn` reproduces the driver state: the racing request's row is staged,
    the INSERT aborts the transaction, and every statement after it raises until
    `rollback()` is called."""
    def winner_creates_the_cart(raw):
        CS.get_or_create(raw, "tokWINNER", email="a@x.com")

    wrapped = AbortingConn(cx, fail_on="INSERT INTO carts",
                           on_fail=winner_creates_the_cart)

    assert CS.get_or_create(wrapped, "tokMINE", email="A@X.com") == "tokWINNER"
    assert wrapped.rollbacks == 1


def test_get_or_create_race_recovery_read_would_raise_without_the_rollback(cx):
    """Anchors the assertion above to the ROLLBACK specifically, not to luck: with
    the transaction left aborted, the recovery SELECT really does raise."""
    wrapped = AbortingConn(cx)
    CS.get_or_create(cx, "tokWINNER", email="a@x.com")
    wrapped.abort()
    with pytest.raises(AbortedTransaction):
        CS.open_token_for_email(wrapped, "a@x.com")


def test_claim_for_checkout_is_atomic_and_a_second_claim_loses(cx):
    """IMPORTANT: `claim_for_checkout` is the ONLY atomic defense against a
    concurrent double-submit minting two orders and two Stripe sessions for one
    cart. Every 409 test in the suite reaches its 409 through the route's
    `checking_out_for_email` pre-check, which short-circuits BEFORE the claim ever
    runs -- so mutating `return cur.rowcount == 1` to `return True`, or dropping
    `AND status='open'`, left all 68 cart tests green.

    This test bypasses the route entirely and claims the cart directly, so the
    conditional UPDATE and its rowcount are the only things under test."""
    CS.get_or_create(cx, "tok1", email="a@x.com")
    CS.add_item(cx, "tok1", "brain-boost", qty=2)

    assert CS.claim_for_checkout(cx, "tok1") is True
    before = cx.execute(
        "SELECT status, claimed_at, updated_at FROM carts WHERE token=?",
        ("tok1",)).fetchone()

    # The concurrent double-submit: same token, cart already claimed.
    assert CS.claim_for_checkout(cx, "tok1") is False

    after = cx.execute(
        "SELECT status, claimed_at, updated_at FROM carts WHERE token=?",
        ("tok1",)).fetchone()
    # The losing claim must not have re-stamped the row either -- re-claiming would
    # slide claimed_at forward and hide the original claim from staleness recovery.
    assert after == before


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
