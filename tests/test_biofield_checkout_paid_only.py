import json
import app
from dashboard import qbo_billing


def _client():
    return app.app.test_client()


def test_biofield_checkout_creates_no_qbo_invoice(monkeypatch):
    """Guard (mutation-style): checkout must NOT POST an invoice to QBO."""
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


def test_biofield_checkout_persists_qbo_lines(monkeypatch):
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
    cx = sqlite3.connect(app.LOG_DB); cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, ref)
    assert row is not None
    payload = json.loads(row["qbo_lines_json"])
    assert payload["lines"]  # line-faithful payload was stored
