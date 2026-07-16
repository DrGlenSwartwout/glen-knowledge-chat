import json
import sqlite3

import app
from dashboard import orders as O
from dashboard import qbo_billing


def _client():
    return app.app.test_client()


def _isolate_db(monkeypatch, tmp_path):
    """Point app.LOG_DB at a throwaway sqlite file (never touching the real local
    chat_log.db) and init the orders schema, mirroring
    tests/test_biofield_checkout_paid_only.py's _isolate_db fixture (Task 3)."""
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    cx = sqlite3.connect(db)
    try:
        O.init_orders_table(cx)
        cx.commit()
    finally:
        cx.close()
    return db


# Real active (non info_only) catalog slug from data/products.json (same source
# used by Task 4's tests/test_begin_checkout_paid_only.py).
ADDON_SLUG = "brain-boost"
ADDON_PRICE_CENTS = 6997  # data/products.json price_cents for ADDON_SLUG


def test_concierge_add_appends_to_order_no_qbo_call(monkeypatch, tmp_path):
    """Guard (mutation-style): concierge add must not call QBO pre-payment; it
    must append the add-on line to the pending order's qbo_lines_json and
    re-total, instead of hitting a live QBO invoice."""
    db = _isolate_db(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise AssertionError("concierge add must not call QBO pre-payment")
    monkeypatch.setattr(qbo_billing, "add_invoice_line", boom)
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)

    # seed a pending order with a checkout_ref token + one line
    ref = "tok-concierge-1"
    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    O.upsert_order(cx, source="funnel", external_ref=ref, email="e@b.com", total_cents=8000)
    O.set_order_qbo_lines(cx, ref, {"lines": [{"name": "Base", "amount": 80.0, "qty": 1}],
                                    "discount_cents": 0, "tax_cents": 0})
    cx.close()

    r = _client().post("/begin/concierge/add",
                       json={"slug": ADDON_SLUG, "invoice_id": ref})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body.get("ok") is True

    cx2 = sqlite3.connect(db)
    cx2.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx2, ref)
    lines = json.loads(row["qbo_lines_json"])["lines"]
    assert len(lines) == 2  # the add-on line was appended
    assert row["total_cents"] == 8000 + ADDON_PRICE_CENTS
    assert row["email"] == "e@b.com"  # unrelated order fields must not be clobbered


def test_concierge_add_returns_404_for_unknown_order(monkeypatch, tmp_path):
    _isolate_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "is_member", lambda sid, email: True)
    r = _client().post("/begin/concierge/add",
                       json={"slug": ADDON_SLUG, "invoice_id": "no-such-ref"})
    assert r.status_code == 404
