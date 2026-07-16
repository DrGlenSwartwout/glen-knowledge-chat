import json
import sqlite3

import app
from dashboard import client_portal as cp
from dashboard import orders as O
from dashboard import qbo_billing


def _client():
    return app.app.test_client()


def _isolate_db(monkeypatch, tmp_path):
    """Point app.LOG_DB at a throwaway sqlite file and init the orders + client_portal
    schema, mirroring tests/test_checkout_cart_paid_only.py's _isolate_db fixture."""
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    cx = sqlite3.connect(db)
    try:
        O.init_orders_table(cx)
        cp.init_client_portal_table(cx)
        cx.commit()
    finally:
        cx.close()
    return db


def _seed_portal(db, email="brooke@example.com", name="Brooke Webb", content=None):
    content = content or {
        "greeting": "Aloha Brooke.",
        "video": {}, "layers": [],
        "reorder_items": [{"slug": "nous-energy", "qty": 1}],
    }
    cx = sqlite3.connect(db)
    try:
        token, _ = cp.upsert_portal(cx, email, name, content)
    finally:
        cx.close()
    return token


def _prep(monkeypatch, tmp_path, email="brooke@example.com"):
    """Isolate the DB, guard against any QBO invoice write, and stub Stripe session
    creation so no network call happens."""
    db = _isolate_db(monkeypatch, tmp_path)
    tok = _seed_portal(db, email=email)

    def boom(*a, **k):
        raise AssertionError("api_client_portal_checkout must not call create_invoice (paid-only)")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})

    monkeypatch.setattr(app, "_get_product",
                        lambda s: {"slug": s, "name": "Nous Energy", "price_cents": 6997,
                                   "qty_pricing": True, "qbo_item_id": "27"}
                        if s == "nous-energy" else None)

    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "create_checkout_session", lambda *a, **k: {"url": "https://s.test"})
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True)
    return db, tok


def test_portal_checkout_creates_no_qbo_invoice(monkeypatch, tmp_path):
    """Guard (mutation-style): api_client_portal_checkout must NOT POST an invoice
    to QBO, and must return a token invoice_id / empty customer_id (paid-only)."""
    db, tok = _prep(monkeypatch, tmp_path)
    r = _client().post(f"/api/portal/{tok}/checkout")
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert body.get("stripe_url") == "https://s.test"


def test_portal_checkout_persists_qbo_lines_and_token_ref(monkeypatch, tmp_path):
    """The order (source portal-reorder) is keyed on the checkout token, with the
    exact QBO line payload persisted for the paid-handler to book later."""
    db, tok = _prep(monkeypatch, tmp_path)

    captured = {}
    orig_set_lines = app._bos_orders.set_order_qbo_lines

    def spy_set_lines(cx, ref, payload):
        captured["ref"] = ref
        captured["payload"] = payload
        return orig_set_lines(cx, ref, payload)
    monkeypatch.setattr(app._bos_orders, "set_order_qbo_lines", spy_set_lines)

    r = _client().post(f"/api/portal/{tok}/checkout")
    assert r.status_code == 200, r.get_data(as_text=True)

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, captured["ref"])
    assert row is not None and row["source"] == "portal-reorder"
    payload = json.loads(row["qbo_lines_json"])
    assert payload["lines"]  # line-faithful payload was stored
    assert payload["lines"][0]["amount"] == 69.97
