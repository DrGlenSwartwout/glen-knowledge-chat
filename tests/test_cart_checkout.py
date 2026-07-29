import os
# Dummy keys so `import app` (which constructs OpenAI + Pinecone clients at import)
# succeeds under a secretless CI without doppler.
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pc-dummy")

import sqlite3

import pytest

import app
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
