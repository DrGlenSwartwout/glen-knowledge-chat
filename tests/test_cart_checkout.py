import os
# Dummy keys so `import app` (which constructs OpenAI + Pinecone clients at import)
# succeeds under a secretless CI without doppler.
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pc-dummy")

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import app
import dashboard.orders as O
from dashboard import cart_store as CS


@pytest.fixture()
def db(monkeypatch, tmp_path):
    path = str(tmp_path / "log.db")
    monkeypatch.setattr(app, "LOG_DB", path)
    monkeypatch.setattr(app, "_PORTAL_CART_ENABLED", True)
    cx = sqlite3.connect(path)
    CS.init_cart_tables(cx)
    cx.close()
    monkeypatch.setattr(
        app, "_get_product",
        lambda slug: {"slug": slug, "name": "Brain Boost", "price_cents": 6997}
        if slug in ("brain-boost", "wholomega") else None)
    return path


@pytest.fixture()
def client():
    return app.app.test_client()


ADDRESS = {"name": "A B", "street": "1 Main", "city": "Hilo",
           "state": "HI", "zip": "96720", "country": "US"}


def _stub_checkout(monkeypatch, seen):
    """Records every keyword the route is supposed to wire through -- not just
    email/cart. `ship`, `points_to_redeem_cents` and `referral_code` all feed real
    money math (shipping/tax, redeemed-points discount, referral commission) and a
    wiring bug in any of them would be invisible if the stub didn't capture it."""
    def fake(email, cart, *, ship, points_to_redeem_cents=0, referral_code=None):
        seen.append({"email": email, "cart": cart, "ship": ship,
                     "points_to_redeem_cents": points_to_redeem_cents,
                     "referral_code": referral_code})
        return {"out": {"invoice_id": "ref123", "total": 69.97},
                "stripe_url": "https://stripe.test/session"}
    monkeypatch.setattr(app, "_checkout_cart", fake)


def test_checkout_404_when_flag_off(client, db, monkeypatch):
    monkeypatch.setattr(app, "_PORTAL_CART_ENABLED", False)
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 404


def test_non_member_gets_need_optin(client, db, monkeypatch):
    monkeypatch.setattr(app, "is_member", lambda sid, email: False)
    client.post("/api/cart/add", json={"slug": "brain-boost"})
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 403
    assert r.get_json()["need_optin"] is True


def test_empty_cart_is_refused(client, db, monkeypatch):
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 400
    assert "empty" in r.get_json()["error"].lower()


def test_unavailable_item_blocks_checkout(client, db, monkeypatch):
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    client.post("/api/cart/add", json={"slug": "brain-boost"})
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    monkeypatch.setattr(app, "_get_product", lambda slug: None)
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 400
    assert "no longer available" in r.get_json()["error"].lower()


def test_info_only_item_blocks_checkout(client, db, monkeypatch):
    """IMPORTANT 7: /api/cart/add already rejects info_only slugs; checkout must
    agree, since a cart can carry a line that was added before a product flipped
    to info_only (8 real SKUs carry this flag)."""
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    client.post("/api/cart/add", json={"slug": "brain-boost"})
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    monkeypatch.setattr(
        app, "_get_product",
        lambda slug: {"slug": slug, "name": "Brain Boost", "info_only": True}
        if slug == "brain-boost" else None)
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 400
    assert "no longer available" in r.get_json()["error"].lower()


def test_missing_address_with_no_saved_order_is_refused(client, db, monkeypatch):
    """CRITICAL 2: a blank body must not check out with no address at all -- that
    both ships nowhere and bypasses the US-only guard downstream. `_checkout_cart`
    must never even be reached."""
    calls = []
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "noaddr@x.com")
    monkeypatch.setattr(app, "_checkout_cart", lambda *a, **k: calls.append(1))
    client.post("/api/cart/add", json={"slug": "brain-boost"})
    r = client.post("/api/cart/checkout", json={})
    assert r.status_code == 400
    assert calls == []


def test_member_checkout_merges_anon_cart_and_marks_ordered(client, db, monkeypatch):
    seen = []
    _stub_checkout(monkeypatch, seen)
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)

    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 2})
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")

    r = client.post("/api/cart/checkout", json={
        "address": ADDRESS, "points_to_redeem_cents": 500, "referral_code": "FRIEND10"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["stripe_url"] == "https://stripe.test/session"

    # the cart reached _checkout_cart in the shape _price_cart reads, and the
    # ship/points/referral wiring is not silently dropped on the way through
    assert seen[0]["email"] == "a@x.com"
    assert seen[0]["cart"] == [{"slug": "brain-boost", "qty": 2, "format": ""}]
    assert seen[0]["ship"] == ADDRESS
    assert seen[0]["points_to_redeem_cents"] == 500
    assert seen[0]["referral_code"] == "FRIEND10"

    # the cart is now the member's, and closed
    cx = sqlite3.connect(app.LOG_DB)
    try:
        assert CS.open_token_for_email(cx, "a@x.com") == ""
        row = cx.execute(
            "SELECT status, checkout_ref FROM carts WHERE checkout_ref='ref123'").fetchone()
        assert row[0] == "ordered"
    finally:
        cx.close()

    # and the visitor's next GET starts clean
    assert client.get("/api/cart").get_json()["items"] == []


def test_checkout_error_is_surfaced_not_swallowed(client, db, monkeypatch):
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")

    def boom(*a, **k):
        raise app.CheckoutError("We only ship within the US right now.")
    monkeypatch.setattr(app, "_checkout_cart", boom)

    client.post("/api/cart/add", json={"slug": "brain-boost"})
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 400
    assert "US" in r.get_json()["error"]

    # IMPORTANT 4 corollary: a CheckoutError must release the claim, not strand the
    # cart in 'checking_out' forever.
    cx = sqlite3.connect(app.LOG_DB)
    try:
        assert CS.open_token_for_email(cx, "a@x.com") != ""
    finally:
        cx.close()


def test_no_stripe_url_returns_ok_with_payment_error_and_closes_cart(client, db, monkeypatch):
    """CRITICAL 1: `_checkout_cart` ingests the order BEFORE minting the Stripe URL, so
    a blank `stripe_url` must NOT be a 502 telling the customer to retry (each retry
    would mint a fresh order -- orphans pile up). It must match the sibling
    /reorder/checkout contract: ok:true, the payment_error key, and the cart closed
    so exactly one order exists per attempt."""
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True)
    monkeypatch.setattr(
        app, "_checkout_cart",
        lambda email, cart, **k: {"out": {"invoice_id": "r1", "total": 1.0},
                                  "stripe_url": ""})
    client.post("/api/cart/add", json={"slug": "brain-boost"})
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["stripe_url"] == ""
    assert body["payment_error"] == app._CARD_UNAVAILABLE

    cx = sqlite3.connect(app.LOG_DB)
    try:
        assert CS.open_token_for_email(cx, "a@x.com") == ""
        row = cx.execute(
            "SELECT status FROM carts WHERE checkout_ref='r1'").fetchone()
        assert row[0] == "ordered"
    finally:
        cx.close()


def test_concurrent_checkout_is_refused_with_409(client, db, monkeypatch):
    """IMPORTANT 4: a second checkout for the same cart while the first is mid-flight
    (already claimed) must not reach `_checkout_cart` a second time."""
    calls = []
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    monkeypatch.setattr(app, "_checkout_cart", lambda *a, **k: calls.append(1))

    client.post("/api/cart/add", json={"slug": "brain-boost"})

    # Simulate another request already mid-flight: claim the cart directly, as the
    # route itself would have done for the first (still-running) request.
    cx = sqlite3.connect(app.LOG_DB)
    try:
        token = CS.open_token_for_email(cx, "a@x.com")
        assert token
        assert CS.claim_for_checkout(cx, token) is True
    finally:
        cx.close()

    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 409
    assert calls == []


def test_get_cart_merges_anon_cookie_cart_for_identified_member(client, db, monkeypatch):
    """IMPORTANT 8: GET must show the union the customer will actually be charged for,
    not just the member cart -- otherwise checkout (which merges) charges for lines
    the page never displayed."""
    # Anonymous session adds an item first, getting the rm_cart cookie.
    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 1})

    # The visitor is now identified as a member with their OWN separate open cart.
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    cx = sqlite3.connect(app.LOG_DB)
    try:
        CS.get_or_create(cx, "mem-a", email="a@x.com")
        CS.add_item(cx, "mem-a", "wholomega", qty=5)
    finally:
        cx.close()

    body = client.get("/api/cart").get_json()
    got = {i["slug"]: i["qty"] for i in body["items"]}
    assert got == {"brain-boost": 1, "wholomega": 5}


def test_get_cart_survives_merge_failure_and_shows_member_cart(client, db, monkeypatch):
    """Round-2 NEW IMPORTANT: the read-time merge added for I8 is a WRITE (UPDATE/
    INSERT/DELETE + commit), not a read. If it raises, the GET must degrade to the
    member's own cart, not blow up into an HTML 500."""
    # Anonymous session adds an item first, getting the rm_cart cookie.
    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 1})

    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    cx = sqlite3.connect(app.LOG_DB)
    try:
        CS.get_or_create(cx, "mem-b", email="a@x.com")
        CS.add_item(cx, "mem-b", "wholomega", qty=3)
    finally:
        cx.close()

    def boom(*a, **k):
        raise RuntimeError("merge exploded")
    monkeypatch.setattr(app._cart_store, "merge", boom)

    r = client.get("/api/cart")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    got = {i["slug"]: i["qty"] for i in body["items"]}
    # degraded to the member cart only -- the anon cart was never folded in
    assert got == {"wholomega": 3}


def test_mark_ordered_raising_still_returns_200_with_payment_link(client, db, monkeypatch):
    """Pin IMPORTANT 5: a mark_ordered failure must not turn an already-priced,
    already-charged (or about to be charged) order into a lost payment link."""
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    monkeypatch.setattr(
        app, "_checkout_cart",
        lambda email, cart, **k: {"out": {"invoice_id": "ref-mo-2", "total": 5.0},
                                  "stripe_url": "https://stripe.test/mo"})

    def always_raise(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(CS, "mark_ordered", always_raise)

    client.post("/api/cart/add", json={"slug": "brain-boost"})
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["stripe_url"] == "https://stripe.test/mo"


def test_unexpected_exception_returns_json_500(client, db, monkeypatch):
    """Pin IMPORTANT 6: a non-CheckoutError exception must come back as JSON, not an
    HTML 500 that a fetch().json() client chokes on -- and must not strand the claim."""
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")

    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(app, "_checkout_cart", boom)

    client.post("/api/cart/add", json={"slug": "brain-boost"})
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 500
    assert r.content_type.startswith("application/json")
    assert r.get_json()["ok"] is False

    # the claim must not be left stuck either
    cx = sqlite3.connect(app.LOG_DB)
    try:
        assert CS.open_token_for_email(cx, "a@x.com") != ""
    finally:
        cx.close()


def test_fresh_claim_still_409s(client, db, monkeypatch):
    """Round-2 NEW CRITICAL guard: a claim that is NOT yet stale (well within the
    staleness window) must still be refused as a genuine concurrent checkout, not
    "recovered" out from under the request that is legitimately still running."""
    calls = []
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    monkeypatch.setattr(app, "_checkout_cart", lambda *a, **k: calls.append(1))

    client.post("/api/cart/add", json={"slug": "brain-boost"})

    cx = sqlite3.connect(app.LOG_DB)
    try:
        token = CS.open_token_for_email(cx, "a@x.com")
        assert token
        assert CS.claim_for_checkout(cx, token) is True
    finally:
        cx.close()

    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 409
    assert calls == []


def test_stale_claim_with_an_existing_order_is_closed_not_reopened(client, db, monkeypatch):
    """Round-2 NEW CRITICAL fix: a `checking_out` claim older than the staleness
    window whose order DID get ingested (mark_ordered failed, or the process died
    right after ingest) must be closed, not reopened -- reopening it would let the
    same items be bought a SECOND time."""
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")

    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 2})

    old_claimed_at = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    cx = sqlite3.connect(app.LOG_DB)
    try:
        old_token = CS.open_token_for_email(cx, "a@x.com")
        assert old_token
        assert CS.claim_for_checkout(cx, old_token) is True
        cx.execute("UPDATE carts SET claimed_at=? WHERE token=?", (old_claimed_at, old_token))
        cx.commit()
    finally:
        cx.close()

    # The checkout that claimed it really did ingest an order (created at/after the
    # claim) before dying -- exactly what `_checkout_cart` does before minting Stripe.
    ocx = sqlite3.connect(app.LOG_DB)
    try:
        O.init_orders_table(ocx)
        O.upsert_order(ocx, source="reorder", external_ref="prior-order-1", email="a@x.com",
                       items=[{"slug": "brain-boost", "qty": 2}], total_cents=13994,
                       address=ADDRESS)
    finally:
        ocx.close()

    # A fresh add for THIS new checkout attempt -- the member's old cart is stuck
    # 'checking_out', so /api/cart/add mints a brand new open cart for them.
    client.post("/api/cart/add", json={"slug": "wholomega", "qty": 1})

    seen = []
    _stub_checkout(monkeypatch, seen)
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True

    # the NEW checkout only ever saw the new cart's item -- the stale cart's
    # brain-boost items were never re-priced or re-sent to _checkout_cart
    assert seen[0]["cart"] == [{"slug": "wholomega", "qty": 1, "format": ""}]

    # the old, stale, already-ordered cart ends closed, referencing the order that
    # was already found -- not reopened
    cx = sqlite3.connect(app.LOG_DB)
    try:
        row = cx.execute(
            "SELECT status, checkout_ref FROM carts WHERE token=?", (old_token,)).fetchone()
        assert row[0] == "ordered"
        assert row[1] == "prior-order-1"
    finally:
        cx.close()


def test_stale_claim_with_no_order_is_released_and_checkout_proceeds(client, db, monkeypatch):
    """Round-2 NEW CRITICAL fix, other branch: a `checking_out` claim older than the
    staleness window whose checkout died BEFORE any order was ingested must be
    released back to 'open' so its items are usable again, and this checkout must
    then proceed with them."""
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")

    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 2})

    old_claimed_at = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    cx = sqlite3.connect(app.LOG_DB)
    try:
        old_token = CS.open_token_for_email(cx, "a@x.com")
        assert old_token
        assert CS.claim_for_checkout(cx, old_token) is True
        cx.execute("UPDATE carts SET claimed_at=? WHERE token=?", (old_claimed_at, old_token))
        cx.commit()
    finally:
        cx.close()

    # The orders table exists (as it always does in production) but has no row for
    # this email -- the earlier attempt died before `_checkout_cart` ever got to
    # ingest one. A clean "table exists, no matching row" is what distinguishes this
    # from the lookup-RAISES case covered separately below.
    ocx = sqlite3.connect(app.LOG_DB)
    try:
        O.init_orders_table(ocx)
    finally:
        ocx.close()

    seen = []
    _stub_checkout(monkeypatch, seen)
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True

    # the released cart's original items were used, not lost
    assert seen[0]["cart"] == [{"slug": "brain-boost", "qty": 2, "format": ""}]

    cx = sqlite3.connect(app.LOG_DB)
    try:
        row = cx.execute(
            "SELECT status FROM carts WHERE token=?", (old_token,)).fetchone()
        assert row[0] == "ordered"  # released, then this same checkout closed it
    finally:
        cx.close()


def test_a_failing_order_lookup_keeps_the_claim_rather_than_risking_a_double_charge(
        client, db, monkeypatch):
    """Round-2 fix-round-3: when the order lookup inside stale-claim recovery itself
    RAISES (as distinct from cleanly finding no order), the claim must be left
    exactly as it was -- not released, not closed. Releasing on an unknown-state
    lookup failure would risk reopening a cart whose order actually exists, letting
    the same items be charged a second time. A wrongly-locked customer is
    recoverable (the same recovery re-runs on their next attempt); a double charge
    is not."""
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")

    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 2})

    old_claimed_at = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    cx = sqlite3.connect(app.LOG_DB)
    try:
        old_token = CS.open_token_for_email(cx, "a@x.com")
        assert old_token
        assert CS.claim_for_checkout(cx, old_token) is True
        cx.execute("UPDATE carts SET claimed_at=? WHERE token=?", (old_claimed_at, old_token))
        cx.commit()
    finally:
        cx.close()

    def boom(*a, **k):
        raise RuntimeError("orders db unreachable")
    monkeypatch.setattr(O, "list_orders_by_email", boom)

    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 409

    # neither released nor closed -- exactly as it was before this request
    cx = sqlite3.connect(app.LOG_DB)
    try:
        row = cx.execute(
            "SELECT status, claimed_at FROM carts WHERE token=?", (old_token,)).fetchone()
        assert row[0] == "checking_out"
        assert row[1] == old_claimed_at
    finally:
        cx.close()


def test_mark_ordered_failure_does_not_lock_the_member_out(client, db, monkeypatch):
    """Round-2 NEW CRITICAL, trigger 1 (the one the round-1 concern predicted):
    `mark_ordered` failing on the first checkout must not return a permanent 409 to
    every future checkout attempt for this member. Forces mark_ordered to raise only
    during the FIRST checkout (both the original attempt and its retry -- see
    `_mark_ordered_resilient`), then relies on the stale-claim recovery (with the
    staleness window shrunk to 0 so the test doesn't need to sleep) to un-stick it
    on a later attempt."""
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    monkeypatch.setattr(app, "_CART_CLAIM_STALE_SECONDS", 0)

    def fake_checkout_that_really_ingests(email, cart, **k):
        # Mirrors what the real _checkout_cart does: the order is ingested BEFORE
        # this function returns, regardless of what happens to mark_ordered after.
        ocx = sqlite3.connect(app.LOG_DB)
        try:
            O.init_orders_table(ocx)
            O.upsert_order(ocx, source="reorder", external_ref="ref-lockout-1", email=email,
                           items=cart, total_cents=6997, address=ADDRESS)
        finally:
            ocx.close()
        return {"out": {"invoice_id": "ref-lockout-1", "total": 69.97},
                "stripe_url": "https://stripe.test/lockout"}
    monkeypatch.setattr(app, "_checkout_cart", fake_checkout_that_really_ingests)

    real_mark_ordered = CS.mark_ordered
    calls = {"n": 0}

    def flaky_mark_ordered(cx, token, ref):
        calls["n"] += 1
        if calls["n"] <= 2:   # the first checkout's original attempt + its one retry
            raise RuntimeError("boom")
        return real_mark_ordered(cx, token, ref)
    monkeypatch.setattr(CS, "mark_ordered", flaky_mark_ordered)

    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 1})
    cx = sqlite3.connect(app.LOG_DB)
    try:
        first_token = CS.open_token_for_email(cx, "a@x.com")
        assert first_token
    finally:
        cx.close()

    r1 = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r1.status_code == 200
    body1 = r1.get_json()
    assert body1["ok"] is True
    assert body1["stripe_url"] == "https://stripe.test/lockout"

    # The cart is still stuck 'checking_out' -- mark_ordered never succeeded, and
    # every future checkout for this member would 409 forever without recovery.
    cx = sqlite3.connect(app.LOG_DB)
    try:
        row = cx.execute(
            "SELECT status FROM carts WHERE token=?", (first_token,)).fetchone()
        assert row[0] == "checking_out"
    finally:
        cx.close()

    # The member tries again with something new to buy.
    client.post("/api/cart/add", json={"slug": "wholomega", "qty": 1})

    r2 = client.post("/api/cart/checkout", json={"address": ADDRESS})
    # The critical assertion: NOT a 409 -- the member is not locked out forever.
    assert r2.status_code != 409
    assert r2.status_code == 200
    assert r2.get_json()["ok"] is True
