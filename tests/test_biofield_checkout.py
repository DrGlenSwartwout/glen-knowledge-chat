# tests/test_biofield_checkout.py
import sqlite3
import app as appmod
from dashboard import biofield_store, points


def _setup(monkeypatch, tmp_path):
    """Stub the money side: consent, QBO, Stripe, order ingest, LOG_DB."""
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(appmod, "LOG_DB", db)
    # consent gate satisfied
    monkeypatch.setattr(appmod, "is_member", lambda sid, email: True)
    monkeypatch.setattr(appmod, "_STRIPE_ACTIVE", True, raising=False)
    monkeypatch.setattr(appmod, "_QBO_PAYMENTS_ACTIVE", True, raising=False)

    cap = {}

    import dashboard.qbo_billing as _qb
    monkeypatch.setattr(_qb, "find_or_create_customer", lambda *a, **k: {"Id": "C1"})

    def fake_invoice(cust, lines, **kw):
        cap["lines"] = lines
        cap["invoice_kw"] = kw
        return {"Id": "INVB", "SyncToken": "0", "DocNumber": "9", "TotalAmt": 300.0}
    monkeypatch.setattr(_qb, "create_invoice", fake_invoice)
    monkeypatch.setattr(_qb, "get_invoice_pay_link", lambda inv: "")

    import dashboard.stripe_pay as _sp
    def fake_session(amount_cents, *, customer_email, description, metadata,
                     success_url, cancel_url, save_card=False):
        cap["stripe_amount"] = amount_cents
        cap["stripe_metadata"] = metadata
        cap["stripe_email"] = customer_email
        cap["stripe_success"] = success_url
        return {"id": "cs_test", "url": "https://stripe/biofield"}
    monkeypatch.setattr(_sp, "create_checkout_session", fake_session)
    monkeypatch.setattr(appmod.stripe_pay, "create_checkout_session", fake_session, raising=False)

    monkeypatch.setattr(appmod, "_ingest_order", lambda **kw: cap.setdefault("order", kw))
    return cap, db


def test_biofield_checkout_creates_300_session(monkeypatch, tmp_path):
    cap, db = _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("BIOFIELD_CHECKOUT_ENABLED", "1")
    c = appmod.app.test_client()
    r = c.post("/biofield/checkout", json={"email": "buyer@x.com", "name": "B"})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert body["stripe_url"] == "https://stripe/biofield"
    # $300 charged
    assert cap["stripe_amount"] == 30000
    md = cap["stripe_metadata"]
    assert md["kind"] == "biofield"
    assert md["email"] == "buyer@x.com"
    # paid-only: invoice_id now carries the checkout_ref token (frontend compat),
    # not a real QBO invoice Id — no invoice is created at checkout.
    assert md.get("invoice_id")
    assert md.get("invoice_id") == body["invoice_id"]
    assert cap["stripe_success"].endswith("/begin/checkout-return?session_id={CHECKOUT_SESSION_ID}")
    # one QBO line, the service item (persisted via qbo_lines payload, not create_invoice)
    assert len(cap["order"]["items"]) == 1
    assert "Biofield" in cap["order"]["items"][0]["name"]
    assert abs(cap["order"]["total_cents"] - 30000) < 1
    # recorded order, no shipping for a service item
    assert cap["order"]["source"] == "biofield"
    assert cap["order"]["shipping_cents"] == 0


def test_biofield_points_reduce_price_respecting_floor(monkeypatch, tmp_path):
    cap, db = _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("BIOFIELD_CHECKOUT_ENABLED", "1")
    # seed a points balance large enough to exceed the floor reduction
    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    points.init_points_table(cx)
    points.earn(cx, "buyer@x.com", full_price_cents=2_000_000, earn_pct=0.05, order_ref="seed")
    cx.commit(); cx.close()

    c = appmod.app.test_client()
    r = c.post("/biofield/checkout",
               json={"email": "buyer@x.com", "name": "B", "points_to_redeem_cents": 5000})
    assert r.status_code == 200, r.get_data(as_text=True)
    charged = cap["stripe_amount"]
    # points reduced the price
    assert charged < 30000
    # never below the engine's points floor for a $300 list (43% => 12900)
    floor = int(round(30000 * 0.43))
    assert charged >= floor
    # with 5000 well above floor, the full 5000 applies
    assert charged == 25000
    assert int(cap["stripe_metadata"]["points_redeemed_cents"]) == 5000


def test_biofield_return_seeds_readiness_and_settles_points(monkeypatch, tmp_path):
    cap, db = _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("BIOFIELD_CHECKOUT_ENABLED", "1")

    import dashboard.stripe_pay as _sp
    monkeypatch.setattr(_sp, "get_session", lambda sid: {
        "id": sid, "payment_status": "paid", "amount_total": 30000,
        "payment_intent": "pi_1",
        "metadata": {"kind": "biofield", "email": "buyer@x.com",
                     "invoice_id": "INVB", "customer_id": "C1",
                     "points_redeemed_cents": "0"},
    })
    monkeypatch.setattr(appmod.stripe_pay, "get_session", _sp.get_session, raising=False)

    import dashboard.qbo_billing as _qb
    monkeypatch.setattr(_qb, "record_payment", lambda *a, **k: None)

    # provide an order row for the points settlement path
    fake_order = {"id": 1, "email": "buyer@x.com", "total_cents": 30000,
                  "shipping_cents": 0, "get_cents": 0, "discount_cents": 0,
                  "points_redeemed_cents": 0}
    monkeypatch.setattr(appmod._bos_orders, "find_order_by_external_ref",
                        lambda cx, ref: dict(fake_order))
    monkeypatch.setattr(appmod._bos_orders, "set_order_stripe_pi",
                        lambda cx, oid, pi: None)

    settled = {}
    monkeypatch.setattr(appmod, "_settle_order_points",
                        lambda order, *, order_ref: settled.update(order_ref=order_ref))

    c = appmod.app.test_client()
    r = c.get("/begin/checkout-return?session_id=cs_test")
    assert r.status_code == 302

    cx = sqlite3.connect(db); cx.row_factory = sqlite3.Row
    biofield_store.init_table(cx)
    row = biofield_store.get(cx, "buyer@x.com")
    cx.close()
    assert row is not None
    assert row["paid_at"]
    assert row["paid_via"] == "stripe"
    assert row["order_ref"] == "INVB"
    # points settlement was invoked for this order
    assert settled.get("order_ref") == "INVB"


def test_biofield_checkout_disabled_when_flag_off(monkeypatch, tmp_path):
    cap, db = _setup(monkeypatch, tmp_path)
    monkeypatch.delenv("BIOFIELD_CHECKOUT_ENABLED", raising=False)
    c = appmod.app.test_client()
    r = c.post("/biofield/checkout", json={"email": "buyer@x.com", "name": "B"})
    assert r.status_code in (403, 404)
    assert "stripe_amount" not in cap   # no Stripe session created


def test_biofield_checkout_scalable_tier_charges_100(monkeypatch, tmp_path):
    cap, db = _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("BIOFIELD_CHECKOUT_ENABLED", "1")
    c = appmod.app.test_client()
    r = c.post("/biofield/checkout",
               json={"email": "buyer@x.com", "name": "B", "tier": "scalable"})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert cap["stripe_amount"] == 10000
    assert cap["stripe_metadata"]["tier"] == "scalable"


def test_biofield_checkout_defaults_premium_unchanged(monkeypatch, tmp_path):
    cap, db = _setup(monkeypatch, tmp_path)
    monkeypatch.setenv("BIOFIELD_CHECKOUT_ENABLED", "1")
    c = appmod.app.test_client()
    r = c.post("/biofield/checkout", json={"email": "buyer@x.com", "name": "B"})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert cap["stripe_amount"] == 30000
    assert cap["stripe_metadata"].get("tier", "premium") == "premium"
