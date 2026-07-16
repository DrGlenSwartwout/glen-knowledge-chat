import json
import sqlite3
import app
from dashboard import orders as _orders_mod
from dashboard import qbo_billing


def _client():
    return app.app.test_client()


def _isolate_db(monkeypatch, tmp_path):
    """Point app.LOG_DB at a throwaway sqlite file (never touching the real local
    chat_log.db) and init the orders schema, mirroring tests/test_biofield_checkout.py's
    _setup() -- these tests, unlike that sibling file, don't stub out _ingest_order, so
    the orders table must actually exist on the fresh db."""
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    cx = sqlite3.connect(db)
    try:
        _orders_mod.init_orders_table(cx)
        cx.commit()
    finally:
        cx.close()
    return db


def test_biofield_checkout_creates_no_qbo_invoice(monkeypatch, tmp_path):
    """Guard (mutation-style): checkout must NOT POST an invoice to QBO."""
    _isolate_db(monkeypatch, tmp_path)
    def boom(*a, **k):
        raise AssertionError("biofield_checkout must not call create_invoice (paid-only)")
    monkeypatch.setattr(qbo_billing, "create_invoice", boom)
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(app, "_biofield_enabled", lambda: True)
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True)
    # stub Stripe session creation so no network call
    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "create_checkout_session",
                        lambda *a, **k: {"url": "https://stripe.test/s"})
    r = _client().post("/biofield/checkout", json={"email": "a@b.com", "name": "A"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    # response carries a stable ref token under the invoice_id field (frontend compat)
    assert body.get("invoice_id")


def test_biofield_checkout_charges_subtotal_drops_get(monkeypatch, tmp_path):
    """Money-path fix: biofield is a service line (shipping_cents=0), so the corrected
    charge = total_cents - get_cents + shipping_cents must equal the bare subtotal --
    any GET the pricing engine computed must never be charged to the customer, only
    recorded for remittance. _price_biofield hardcodes ship_to_state=None so the real
    engine never actually produces a nonzero get_cents here; stub a crafted pc so the
    checkout route's own arithmetic is what's under test, not the engine's tax gate."""
    db = _isolate_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "_biofield_enabled", lambda: True)
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True)

    fake_pc = {
        "priced": {"total_cents": 31000, "get_cents": 1000},
        "qbo_lines": [{"name": "Biofield Analysis (Premium)", "amount": 300.0,
                       "qty": 1, "description": "Biofield Analysis (Premium)"}],
        "items_rec": [{"name": "Biofield Analysis (Premium)", "qty": 1,
                       "desc": "Biofield Analysis (Premium)"}],
        "discount_cents": 0,
        "points_redeemed_cents": 0,
        "shipping_cents": 0,
    }
    monkeypatch.setattr(app, "_price_biofield", lambda **kw: fake_pc)

    cap = {}
    import dashboard.stripe_pay as sp
    def fake_session(amount_cents, **kw):
        cap["amount_cents"] = amount_cents
        return {"url": "https://stripe.test/s"}
    monkeypatch.setattr(sp, "create_checkout_session", fake_session)

    r = _client().post("/biofield/checkout", json={"email": "hi@b.com", "name": "H"})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    # 31000 total - 1000 get + 0 shipping = 30000 (bare subtotal); GET must be dropped.
    assert cap["amount_cents"] == 30000
    assert body["total"] == 300.0

    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    row = _orders_mod.find_order_by_external_ref(cx, body["invoice_id"])
    assert row["total_cents"] == 30000


def test_biofield_checkout_persists_qbo_lines(monkeypatch, tmp_path):
    db = _isolate_db(monkeypatch, tmp_path)
    monkeypatch.setattr(qbo_billing, "create_invoice",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no invoice")))
    monkeypatch.setattr(qbo_billing, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})
    monkeypatch.setattr(app, "_biofield_enabled", lambda: True)
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", True)
    import dashboard.stripe_pay as sp
    monkeypatch.setattr(sp, "create_checkout_session", lambda *a, **k: {"url": "u"})
    r = _client().post("/biofield/checkout", json={"email": "b@b.com", "name": "B"})
    ref = r.get_json()["invoice_id"]
    import sqlite3
    from dashboard import orders as O
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, ref)
    assert row is not None
    payload = json.loads(row["qbo_lines_json"])
    assert payload["lines"]  # line-faithful payload was stored
