import sqlite3

import pytest

from dashboard import cart_store as CS
from tests.aborting_conn import AbortingConn


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


def test_merge_is_idempotent_after_folding_into_a_pre_existing_member_cart(cx):
    # Pre-existing member cart
    CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=3)
    CS.add_item(cx, "mem1", "neuroprotect", qty=1)

    # Anonymous cart with overlapping and new items
    CS.get_or_create(cx, "anon1")
    CS.add_item(cx, "anon1", "brain-boost", qty=2)  # lower, higher-wins applies
    CS.add_item(cx, "anon1", "wholomega", qty=1)     # new item

    # Call merge three times
    first = CS.merge(cx, "anon1", "a@x.com")
    second = CS.merge(cx, "anon1", "a@x.com")
    third = CS.merge(cx, "anon1", "a@x.com")

    # All three calls return the same member token
    assert first == second == third == "mem1"

    # Verify final state: higher qty wins, new items included
    got = {i["slug"]: i["qty"] for i in CS.items(cx, "mem1")}
    assert got == {"brain-boost": 3, "neuroprotect": 1, "wholomega": 1}

    # Only one open cart for this email
    count = cx.execute(
        "SELECT COUNT(*) FROM carts WHERE email=? AND status='open'", ("a@x.com",)
    ).fetchone()[0]
    assert count == 1

    # Anon cart is merged with no items
    assert CS.items(cx, "anon1") == []
    row = cx.execute("SELECT status FROM carts WHERE token=?", ("anon1",)).fetchone()
    assert row[0] == "merged"


def test_merge_refuses_a_cart_owned_by_another_member(cx):
    """CRITICAL 3: a shared browser's rm_cart cookie can carry a DIFFERENT member's
    open cart token. Merging it onto the current buyer must be a complete no-op for
    that cart: not reassigned, not folded from, not marked merged -- probed as Bob
    getting charged for Alice's items on a shared browser."""
    CS.get_or_create(cx, "bob_cart", email="bob@x.com")
    CS.add_item(cx, "bob_cart", "brain-boost", qty=1)

    CS.get_or_create(cx, "alice_cart", email="alice@x.com")
    CS.add_item(cx, "alice_cart", "wholomega", qty=2)

    # Bob's browser carries Alice's cookie token (e.g. a shared kiosk). Merging it
    # onto Bob must leave Alice's cart completely untouched and fall through to
    # Bob's own cart.
    surviving = CS.merge(cx, "alice_cart", "bob@x.com")

    assert surviving == "bob_cart"
    assert {i["slug"]: i["qty"] for i in CS.items(cx, "bob_cart")} == {"brain-boost": 1}

    # Alice's cart keeps its items, stays open, and keeps her email
    assert {i["slug"]: i["qty"] for i in CS.items(cx, "alice_cart")} == {"wholomega": 2}
    row = cx.execute(
        "SELECT status, email FROM carts WHERE token=?", ("alice_cart",)).fetchone()
    assert row[0] == "open"
    assert row[1] == "alice@x.com"


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


def test_merge_recovers_when_the_member_cart_appears_mid_flight(cx):
    """Simulate a race where the member cart is created between the read and write."""
    # Create a member cart
    mem_token = CS.get_or_create(cx, "mem1", email="a@x.com")
    CS.add_item(cx, "mem1", "brain-boost", qty=2)

    # Create an anonymous cart
    CS.get_or_create(cx, "anon1")
    CS.add_item(cx, "anon1", "wholomega", qty=1)

    # Monkeypatch open_token_for_email so the first call returns "" (simulating the race
    # where it hasn't found the member cart yet), and subsequent calls return the real token
    call_count = [0]
    original_func = CS.open_token_for_email

    def patched_open_token_for_email(cx, email):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: pretend we don't have a member cart yet (simulating race condition)
            return ""
        # Subsequent calls: return the real member token
        return original_func(cx, email)

    try:
        CS.open_token_for_email = patched_open_token_for_email
        # Call merge - it should handle the race gracefully
        result = CS.merge(cx, "anon1", "a@x.com")
    finally:
        CS.open_token_for_email = original_func

    # Merge should have recovered and returned the real member token
    assert result == mem_token
    # The anonymous items should be merged into the member cart
    got = {i["slug"]: i["qty"] for i in CS.items(cx, mem_token)}
    assert got == {"brain-boost": 2, "wholomega": 1}


def test_merge_race_recovery_survives_an_aborted_transaction(cx):
    """CRITICAL, same root cause as `get_or_create`'s: `merge`'s race self-heal
    recovers by re-reading (`open_token_for_email`) after the claim UPDATE fails.
    On Postgres that failed UPDATE leaves the transaction ABORTED, so the recovery
    read raises `InFailedSqlTransaction` and the anonymous cart's items are LOST
    instead of folded onto the member. sqlite3 keeps the connection usable, which
    is the only reason the pre-existing race test passes.

    `AbortingConn` reproduces the driver state: the racing request's member cart is
    staged, the claim UPDATE aborts the transaction, and every statement after it
    raises until `rollback()`."""
    CS.get_or_create(cx, "anon1")
    CS.add_item(cx, "anon1", "wholomega", qty=1)

    def winner_creates_the_member_cart(raw):
        CS.get_or_create(raw, "mem-race", email="a@x.com")
        CS.add_item(raw, "mem-race", "brain-boost", qty=2)

    wrapped = AbortingConn(cx, fail_on="UPDATE carts SET email=",
                           on_fail=winner_creates_the_member_cart)

    assert CS.merge(wrapped, "anon1", "A@X.com") == "mem-race"
    assert wrapped.rollbacks == 1
    # the anonymous basket was folded on, not stranded
    got = {i["slug"]: i["qty"] for i in CS.items(cx, "mem-race")}
    assert got == {"brain-boost": 2, "wholomega": 1}
    assert cx.execute(
        "SELECT status FROM carts WHERE token=?", ("anon1",)).fetchone()[0] == "merged"
