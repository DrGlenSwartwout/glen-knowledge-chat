"""Tests for the site-wide Records search endpoint /api/console/search.

Records mode of the header search box: look up a specific person / product /
order. Read-only, gated by the console key like /api/people. Mirrors the
LOG_DB / CONSOLE_SECRET monkeypatch pattern in test_access_token_guard.py.
"""

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest


def _app():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app module not importable: {e}")


def _seed(tmp_db):
    from dashboard import orders as O
    with sqlite3.connect(tmp_db) as cx:
        cx.execute(
            "CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, email TEXT, "
            "first_name TEXT, last_name TEXT, phone TEXT, order_count INTEGER DEFAULT 0, "
            "last_order_date TEXT)"
        )
        cx.execute(
            "INSERT INTO people (name, email, first_name, last_name, phone, order_count) "
            "VALUES ('John Doe', 'john@example.com', 'John', 'Doe', '8085551234', 3)"
        )
        cx.execute(
            "INSERT INTO people (name, email, first_name, last_name, order_count) "
            "VALUES ('Jane Smith', 'jane@example.com', 'Jane', 'Smith', 1)"
        )
        O.init_orders_table(cx)
        cx.execute(
            "INSERT INTO orders (created_at, source, external_ref, email, name, total_cents, status) "
            "VALUES ('2026-06-01', 'inhouse', 'ORD-777', 'john@example.com', 'John Doe', 14200, 'paid')"
        )
        cx.commit()


def _setup(monkeypatch, tmp_db):
    app = _app()
    monkeypatch.setattr(app, "LOG_DB", tmp_db)
    monkeypatch.setattr(app, "CONSOLE_SECRET", "testkey")
    # Deterministic product catalog so the test doesn't depend on data/products.json.
    monkeypatch.setattr(
        app._bos_products, "catalog",
        lambda **kw: [{"slug": "nous-energy", "name": "Nous Energy",
                       "ingredients": ["spirit minerals"], "price_cents": 4997}],
    )
    _seed(tmp_db)
    return app


def _get(app, q, key="testkey"):
    headers = {"X-Console-Key": key} if key else {}
    return app.app.test_client().get(f"/api/console/search?q={q}", headers=headers)


def test_requires_key(monkeypatch, tmp_db):
    app = _setup(monkeypatch, tmp_db)
    resp = _get(app, "john", key=None)
    assert resp.status_code == 401


def test_empty_query_returns_empty_groups(monkeypatch, tmp_db):
    app = _setup(monkeypatch, tmp_db)
    resp = _get(app, "")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {"people": [], "products": [], "orders": []}


def test_person_match(monkeypatch, tmp_db):
    app = _setup(monkeypatch, tmp_db)
    body = _get(app, "john").get_json()
    assert any(p["title"] == "John Doe" for p in body["people"])
    person = next(p for p in body["people"] if p["title"] == "John Doe")
    assert person["url"].startswith("/console?pq=")
    assert "john%40example.com" in person["url"]  # email url-encoded


def test_order_match_by_ref(monkeypatch, tmp_db):
    app = _setup(monkeypatch, tmp_db)
    body = _get(app, "ord-777").get_json()
    assert body["orders"], "expected an order match"
    order = body["orders"][0]
    assert order["title"] == "ORD-777"
    assert order["url"].startswith("/console/orders?q=")
    assert "$142" in order["subtitle"]


def test_product_match(monkeypatch, tmp_db):
    app = _setup(monkeypatch, tmp_db)
    body = _get(app, "nous").get_json()
    assert any(p["title"] == "Nous Energy" for p in body["products"])
    prod = next(p for p in body["products"] if p["title"] == "Nous Energy")
    assert prod["url"].startswith("/console/products?q=")


def test_product_match_by_ingredient(monkeypatch, tmp_db):
    app = _setup(monkeypatch, tmp_db)
    body = _get(app, "spirit minerals").get_json()
    assert any(p["title"] == "Nous Energy" for p in body["products"])


def test_types_filter_scopes_results(monkeypatch, tmp_db):
    app = _setup(monkeypatch, tmp_db)
    resp = app.app.test_client().get(
        "/api/console/search?q=john&types=orders",
        headers={"X-Console-Key": "testkey"},
    )
    body = resp.get_json()
    assert body["people"] == []   # people not requested
    assert body["orders"]         # john matches the order email
