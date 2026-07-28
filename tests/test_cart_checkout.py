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
    def fake(email, cart, *, ship, points_to_redeem_cents=0, referral_code=None):
        seen.append({"email": email, "cart": cart})
        return {"out": {"invoice_id": "ref123", "total": 69.97},
                "stripe_url": "https://stripe.test/session"}
    monkeypatch.setattr(app, "_checkout_cart", fake)


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


def test_member_checkout_merges_anon_cart_and_marks_ordered(client, db, monkeypatch):
    seen = []
    _stub_checkout(monkeypatch, seen)
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)

    client.post("/api/cart/add", json={"slug": "brain-boost", "qty": 2})
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")

    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["stripe_url"] == "https://stripe.test/session"

    # the cart reached _checkout_cart in the shape _price_cart reads
    assert seen[0]["email"] == "a@x.com"
    assert seen[0]["cart"] == [{"slug": "brain-boost", "qty": 2, "format": ""}]

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


def test_no_stripe_url_is_an_error_not_a_silent_success(client, db, monkeypatch):
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    monkeypatch.setattr(app, "_cart_email", lambda: "a@x.com")
    monkeypatch.setattr(
        app, "_checkout_cart",
        lambda email, cart, **k: {"out": {"invoice_id": "r1", "total": 1.0},
                                  "stripe_url": ""})
    client.post("/api/cart/add", json={"slug": "brain-boost"})
    r = client.post("/api/cart/checkout", json={"address": ADDRESS})
    assert r.status_code == 502
    assert r.get_json()["ok"] is False
