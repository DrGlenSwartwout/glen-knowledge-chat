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
    monkeypatch.setattr(app, "_PORTAL_CART_ENABLED", False)
    assert client.get("/api/cart").status_code == 404
    assert client.post("/api/cart/add", json={"slug": "brain-boost"}).status_code == 404


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
