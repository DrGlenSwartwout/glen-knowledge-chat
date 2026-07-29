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
    # a real-shaped catalog entry, pinned so $DATA_DIR cannot strip it
    monkeypatch.setattr(
        app, "_get_product",
        lambda slug: {"slug": "brain-boost", "name": "Brain Boost",
                      "price_cents": 6997} if slug == "brain-boost" else None)
    return path


@pytest.fixture()
def client():
    return app.app.test_client()


def test_routes_404_when_flag_off(monkeypatch, client, db):
    """All FOUR cart routes, not three. PORTAL_CART_ENABLED shipping OFF is the
    claim this whole PR rests on, so every route that reads or writes a cart is
    pinned here -- /api/cart/set-qty's guard was the one a deletion survived."""
    monkeypatch.setattr(app, "_PORTAL_CART_ENABLED", False)
    assert client.get("/api/cart").status_code == 404
    assert client.post("/api/cart/add", json={"slug": "brain-boost"}).status_code == 404
    assert client.post("/api/cart/set-qty",
                       json={"slug": "brain-boost", "qty": 3}).status_code == 404
    assert client.post("/api/cart/checkout", json={}).status_code == 404


def test_info_only_line_is_reported_unavailable(client, db, monkeypatch):
    """The badge must agree with the checkout error: `api_cart_checkout` refuses
    info_only lines with "no longer available", so `_cart_payload` reporting them
    as available would show a green line the customer cannot actually buy."""
    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 1})
    monkeypatch.setattr(
        app, "_get_product",
        lambda slug: {"slug": "brain-boost", "name": "Brain Boost",
                      "info_only": True} if slug == "brain-boost" else None)
    body = client.get("/api/cart").get_json()
    assert len(body["items"]) == 1
    assert body["items"][0]["available"] is False


def test_empty_cart_for_a_new_visitor(client, db):
    r = client.get("/api/cart")
    assert r.status_code == 200
    assert r.get_json() == {"ok": True, "items": [], "count": 0}


def test_add_sets_the_cookie_and_persists(client, db):
    r = client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 2})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    assert "rm_cart=" in r.headers.get("Set-Cookie", "")

    r2 = client.get("/api/cart")
    body = r2.get_json()
    assert body["count"] == 2
    assert body["items"][0]["slug"] == "brain-boost"
    assert body["items"][0]["name"] == "Brain Boost"
    assert body["items"][0]["available"] is True


def test_add_rejects_an_unknown_slug(client, db):
    r = client.post("/api/cart/add", json={"slug": "no-such-product"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_add_does_not_require_membership(client, db):
    """Anonymous adds are the whole point -- no need_optin here, only at checkout."""
    r = client.post("/api/cart/add", json={"slug": "brain-boost"})
    assert r.status_code == 200


def test_set_qty_updates_then_removes(client, db):
    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 5})
    client.post("/api/cart/set-qty", json={"slug": "brain-boost", "qty": 2})
    assert client.get("/api/cart").get_json()["count"] == 2
    client.post("/api/cart/set-qty", json={"slug": "brain-boost", "qty": 0})
    assert client.get("/api/cart").get_json()["items"] == []


def test_unavailable_item_is_flagged_not_dropped(client, db, monkeypatch):
    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 1})
    monkeypatch.setattr(app, "_get_product", lambda slug: None)
    body = client.get("/api/cart").get_json()
    assert len(body["items"]) == 1
    assert body["items"][0]["available"] is False


def _cookie_token(client):
    """The current rm_cart cookie value, however this Werkzeug version exposes it."""
    c = client.get_cookie("rm_cart")
    return c.value if c else ""


def test_get_cart_ignores_a_cart_that_was_already_ordered(client, db):
    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 1})
    token = _cookie_token(client)
    assert token

    cx = sqlite3.connect(db)
    CS.mark_ordered(cx, token, "checkout-ref-1")
    cx.close()

    body = client.get("/api/cart").get_json()
    assert body["items"] == []
    assert body["count"] == 0


def test_set_qty_cannot_mutate_an_ordered_cart(client, db):
    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 1})
    token = _cookie_token(client)
    assert token

    cx = sqlite3.connect(db)
    CS.mark_ordered(cx, token, "checkout-ref-2")
    cx.close()

    client.post("/api/cart/set-qty", json={"slug": "brain-boost", "qty": 9})

    cx = sqlite3.connect(db)
    row = cx.execute(
        "SELECT qty FROM cart_items WHERE token=? AND slug=?", (token, "brain-boost")
    ).fetchone()
    cx.close()
    assert row is not None
    assert row[0] == 1  # unchanged -- the ordered cart's record of what was bought


def test_add_after_checkout_starts_a_new_cart_and_recookies(client, db):
    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 1})
    old_token = _cookie_token(client)
    assert old_token

    cx = sqlite3.connect(db)
    CS.mark_ordered(cx, old_token, "checkout-ref-3")
    cx.close()

    r = client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 3})
    assert r.status_code == 200
    set_cookie = r.headers.get("Set-Cookie", "")
    assert "rm_cart=" in set_cookie

    new_token = _cookie_token(client)
    assert new_token
    assert new_token != old_token

    body = client.get("/api/cart").get_json()
    assert body["count"] == 3
    assert len(body["items"]) == 1
    assert body["items"][0]["slug"] == "brain-boost"

    cx = sqlite3.connect(db)
    old_items = cx.execute(
        "SELECT qty FROM cart_items WHERE token=? AND slug=?", (old_token, "brain-boost")
    ).fetchall()
    cx.close()
    assert old_items == [(1,)]  # the ordered cart's line is untouched


# --- IMPORTANT (money): cross-account cart access via the rm_cart cookie ---------
#
# /api/cart/add writes an IDENTIFIED member's cart token into the browser cookie,
# and `_cart_open_token` honoured that cookie for any later visitor. Probed: Bob,
# anonymous on Alice's browser, read her cart, rewrote a quantity to 9 and added a
# line -- and Alice is the one who gets charged. `cart_store.merge` was hardened
# against exactly this in an earlier round; the read/write surface was not. This
# codebase explicitly models shared household devices, so this is the normal case,
# not an exotic one.


@pytest.fixture()
def alices_browser(client, db, monkeypatch):
    """Alice, identified, fills a cart on this browser -- leaving her cart token in
    the rm_cart cookie. Returns (client, alice_token, set_visitor) where
    `set_visitor(email)` becomes the NEXT person on the same browser."""
    holder = {"email": "alice@x.com"}
    monkeypatch.setattr(app, "_cart_email", lambda: holder["email"])
    monkeypatch.setattr(
        app, "_get_product",
        lambda slug: {"slug": slug, "name": slug.title(), "price_cents": 6997}
        if slug in ("brain-boost", "wholomega") else None)

    r = client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 2})
    assert r.status_code == 200
    alice_token = _cookie_token(client)
    assert alice_token

    cx = sqlite3.connect(db)
    try:
        assert cx.execute("SELECT email FROM carts WHERE token=?",
                          (alice_token,)).fetchone()[0] == "alice@x.com"
    finally:
        cx.close()

    def set_visitor(email):
        holder["email"] = email

    return client, alice_token, set_visitor


def _alice_lines(db_path, token):
    cx = sqlite3.connect(db_path)
    try:
        return {r[0]: r[1] for r in cx.execute(
            "SELECT slug, qty FROM cart_items WHERE token=?", (token,)).fetchall()}
    finally:
        cx.close()


def test_anonymous_visitor_cannot_read_the_members_cart_from_the_cookie(
        alices_browser, db):
    client, alice_token, set_visitor = alices_browser
    set_visitor("")                      # Bob, anonymous, same browser
    body = client.get("/api/cart").get_json()
    assert body == {"ok": True, "items": [], "count": 0}


def test_anonymous_visitor_cannot_set_qty_on_the_members_cart(alices_browser, db):
    client, alice_token, set_visitor = alices_browser
    set_visitor("")
    r = client.post("/api/cart/set-qty", json={"slug": "brain-boost", "qty": 9})
    assert r.status_code == 200
    assert r.get_json()["items"] == []
    assert _alice_lines(db, alice_token) == {"brain-boost": 2}   # not rewritten to 9


def test_anonymous_visitor_cannot_add_to_the_members_cart(alices_browser, db):
    client, alice_token, set_visitor = alices_browser
    set_visitor("")
    r = client.post("/api/cart/add", json={"slug": "wholomega", "qty": 1})
    assert r.status_code == 200

    # Bob got a cart of his own, and Alice's is untouched.
    bob_token = _cookie_token(client)
    assert bob_token and bob_token != alice_token
    assert _alice_lines(db, alice_token) == {"brain-boost": 2}
    assert _alice_lines(db, bob_token) == {"wholomega": 1}
    assert r.get_json()["count"] == 1


def test_a_different_identified_member_cannot_touch_the_cookie_cart(alices_browser, db):
    """Not only anonymous visitors: Bob signed in on Alice's browser must get his
    OWN cart, not hers."""
    client, alice_token, set_visitor = alices_browser
    set_visitor("bob@x.com")

    assert client.get("/api/cart").get_json()["items"] == []

    client.post("/api/cart/set-qty", json={"slug": "brain-boost", "qty": 9})
    assert _alice_lines(db, alice_token) == {"brain-boost": 2}

    client.post("/api/cart/add", json={"slug": "wholomega", "qty": 1})
    bob_token = _cookie_token(client)
    assert bob_token != alice_token
    assert _alice_lines(db, alice_token) == {"brain-boost": 2}

    cx = sqlite3.connect(db)
    try:
        assert cx.execute("SELECT email FROM carts WHERE token=?",
                          (bob_token,)).fetchone()[0] == "bob@x.com"
    finally:
        cx.close()
