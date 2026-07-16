"""Reorder cart v1 — magic-link auth, items API, multi-item checkout.

QBO + Stripe + tax are stubbed; the order recording + auth + item resolution run
for real against a tmp LOG_DB.
"""
import importlib
import json
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import begin_funnel


def _app():
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("app")
    except Exception as e:
        pytest.skip(f"app not importable: {e}")


CATALOG = {"default_price_cents": 6997, "products": {
    "terrain-restore": {"name": "Terrain Restore", "price_cents": 6997, "qbo_item_id": "Q1"},
    "brain-cleanse":   {"name": "Brain Cleanse",   "price_cents": 5997, "qbo_item_id": "Q2"},
}}


@pytest.fixture
def app_db(monkeypatch, tmp_path):
    app = _app()
    db = str(tmp_path / "chat_log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    monkeypatch.setattr(app, "_PRODUCTS", CATALOG)
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True)
    with sqlite3.connect(db) as cx:
        cx.execute("CREATE TABLE auth_tokens (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                   "token_hash TEXT, email TEXT, purpose TEXT, created_at TEXT, "
                   "expires_at TEXT, consumed_at TEXT, extra TEXT)")
        app._bos_orders.init_orders_table(cx)
        cx.commit()
    # stub network: QBO + Stripe + tax + shipping
    from dashboard import qbo_billing as qb, stripe_pay, tax as taxmod
    monkeypatch.setattr(qb, "find_or_create_customer", lambda email, name="": {"Id": "C1"})
    def boom(*a, **k):
        raise AssertionError("_checkout_cart must not call create_invoice (paid-only)")
    monkeypatch.setattr(qb, "create_invoice", boom)
    # _checkout_cart is paid-only (QBO Stage 2): no invoice is created at checkout time --
    # the exact QBO line/discount payload is persisted via set_order_qbo_lines instead.
    # last_payload mirrors the old mk.last_lines capture point (a plain function attr, so
    # callers can read it back the same way after the checkout POST).
    captured = {}
    def _capture_qbo_lines(cx, ref, payload):
        captured["payload"] = payload
        _capture_qbo_lines.last_payload = payload
    monkeypatch.setattr(app._bos_orders, "set_order_qbo_lines", _capture_qbo_lines)
    monkeypatch.setattr(taxmod, "compute_get_cents", lambda *a, **k: 0)
    monkeypatch.setattr(stripe_pay, "create_checkout_session",
                        lambda *a, **k: {"id": "cs_1", "url": "https://stripe.test/pay"})
    monkeypatch.setattr(app, "_shipping_for_cart", lambda *a, **k: 0)
    return app, db, _capture_qbo_lines


def _seed_order(app, db, email, items, created_at):
    with sqlite3.connect(db) as cx:
        cx.execute("INSERT INTO orders (created_at, source, external_ref, channel, email, "
                   "items_json, total_cents, address_json) VALUES (?,?,?,?,?,?,?,?)",
                   (created_at, "funnel", created_at, "retail", email,
                    json.dumps(items), 0, json.dumps({"state": "HI", "name": "Pat"})))
        cx.commit()


def _authed_client(app, db, email):
    """Mint a reorder token, redeem it, return a test client carrying the cookie."""
    c = app.app.test_client()
    tok = "tok-" + email.replace("@", "-")
    now = app._now_utc()
    with sqlite3.connect(db) as cx:
        begin_funnel.init_journey_tables(cx)
        begin_funnel.record_unlock(cx, session_id="sess-reorder-test", trigger="tos",
                                   email=email, tos=True)
        cx.execute("INSERT INTO auth_tokens (token_hash, email, purpose, created_at, expires_at) "
                   "VALUES (?,?,?,?,?)",
                   (app._hash_token(tok), email, "reorder", now.isoformat(),
                    (now + timedelta(minutes=15)).isoformat()))
        cx.commit()
    # GET only confirms (mail scanners prefetch it); the POST signs in.
    r = c.post(f"/reorder/auth/{tok}")
    assert r.status_code == 302
    return c


# ── auth ──────────────────────────────────────────────────────────────────────

def test_request_always_200(app_db):
    app, db, _ = app_db
    r = app.app.test_client().post("/reorder/request", json={"email": "anyone@x.com"})
    assert r.status_code == 200 and r.get_json()["ok"] is True


def test_items_requires_login(app_db):
    app, db, _ = app_db
    assert app.app.test_client().get("/api/reorder/items").status_code == 401


def test_invalid_token_rejected(app_db):
    app, db, _ = app_db
    assert app.app.test_client().get("/reorder/auth/nope").status_code == 400


def test_auth_then_items(app_db):
    app, db, _ = app_db
    _seed_order(app, db, "c@x.com", [{"name": "Terrain Restore", "qty": 2}], "2026-06-01")
    c = _authed_client(app, db, "c@x.com")
    d = c.get("/api/reorder/items").get_json()
    assert [i["name"] for i in d["items"]] == ["Terrain Restore"]
    assert d["items"][0]["available"] and d["items"][0]["slug"] == "terrain-restore"


# ── items resolution ──────────────────────────────────────────────────────────

def test_items_scope_and_dedupe(app_db):
    app, db, _ = app_db
    _seed_order(app, db, "c@x.com", [{"name": "Brain Cleanse", "qty": 1}], "2026-05-01")
    _seed_order(app, db, "c@x.com", [{"name": "Terrain Restore", "qty": 2},
                                     {"name": "Discontinued Thing", "qty": 1}], "2026-06-01")
    c = _authed_client(app, db, "c@x.com")
    last = c.get("/api/reorder/items?scope=last").get_json()["items"]
    assert sorted(i["name"] for i in last) == ["Discontinued Thing", "Terrain Restore"]
    disc = [i for i in last if i["name"] == "Discontinued Thing"][0]
    assert disc["available"] is False and disc["slug"] is None
    alln = c.get("/api/reorder/items?scope=all").get_json()["items"]
    assert "Brain Cleanse" in [i["name"] for i in alln]  # earlier order included via scope=all


# ── checkout ──────────────────────────────────────────────────────────────────

def test_checkout_builds_invoice_and_records(app_db):
    """Engine contract: qty-2 terrain-restore (6997) + qty-1 brain-cleanse (5997) ->
    total_months=3 -> volume_pct(3)=17.5%. QBO lines carry LIST amounts (the discount
    rides discount_cents on the invoice, not the per-line price)."""
    app, db, mk = app_db
    _seed_order(app, db, "c@x.com", [{"name": "Terrain Restore", "qty": 1}], "2026-06-01")
    c = _authed_client(app, db, "c@x.com")
    r = c.post("/reorder/checkout", json=[{"slug": "terrain-restore", "qty": 2},
                                          {"slug": "brain-cleanse", "qty": 1}])
    d = r.get_json()
    assert d["ok"] and d["stripe_url"] == "https://stripe.test/pay"
    # paid-only: no real QBO customer/invoice at checkout time
    assert d["customer_id"] == ""
    # QBO line payload built with both lines + correct quantities, at LIST price
    lines = {l["name"]: l for l in mk.last_payload["lines"]}
    assert lines["Terrain Restore"]["qty"] == 2 and lines["Terrain Restore"]["amount"] == 69.97
    assert lines["Brain Cleanse"]["qty"] == 1 and lines["Brain Cleanse"]["amount"] == 59.97
    # order recorded as a reorder, keyed on the checkout token
    with sqlite3.connect(db) as cx:
        cx.row_factory = sqlite3.Row
        o = cx.execute("SELECT * FROM orders WHERE source='reorder'").fetchone()
    assert o and o["external_ref"] == d["invoice_id"]
    # total_cents now reflects the engine-priced total (list minus the real volume
    # discount) rather than the old invoice-mock's naive undiscounted line sum.
    # (set_order_qbo_lines itself is mocked above to capture the payload without
    # touching the DB, so qbo_lines_json on the row is not asserted here.)
    assert o["total_cents"] == int(round(d["total"] * 100))


def test_checkout_empty_or_unavailable_400(app_db):
    app, db, _ = app_db
    c = _authed_client(app, db, "c@x.com")
    assert c.post("/reorder/checkout", json=[]).status_code == 400
    assert c.post("/reorder/checkout", json=[{"slug": "ghost", "qty": 1}]).status_code == 400


def test_checkout_requires_login(app_db):
    app, db, _ = app_db
    assert app.app.test_client().post("/reorder/checkout", json=[{"slug": "terrain-restore", "qty": 1}]).status_code == 401
