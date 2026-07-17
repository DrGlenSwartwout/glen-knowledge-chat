"""QBO paid-only Stage 4, Task 4: dashboard.dropship_checkout.build_client_order --
patient-paid (dispensary) checkout, ships to the patient, practitioner margin
credited on payment.

Unlike wholesale/dropship, this flow's discount was ALREADY resolved before
create_invoice (discount_cents=redeem_cents + ship_credit_applied) -- a straight
Pattern-I conversion, no redeem reorder: a fresh ``checkout_ref`` token is
minted, NO QBO invoice/customer is created at checkout time, and a
line-faithful ``qbo_payload`` is returned for the route to persist -- the
return-handler (already wired for kind='client', Task 1) books a real Sales
Receipt from it once payment is confirmed.

Two layers:
  - unit tests on ``dropship_checkout.build_client_order`` directly (mirrors
    tests/test_client_order.py's fixture shape) for the guard + discount fidelity
    + charge math + preserved points/ship-credit/margin bookkeeping;
  - one route-level test (build_client_order mocked) confirming the persistence
    wiring in ``api_client_checkout`` -- qbo_lines_json, get_cents, and the order
    keyed on the token with source "dispensary".
"""

import json
import sqlite3

import app
from dashboard import dropship_checkout as dc
from dashboard import orders as O
from dashboard import qbo_billing


def _boom(*a, **k):
    raise AssertionError("build_client_order must not touch QBO invoicing (paid-only)")


def _stub_common(monkeypatch, get_cents=0):
    monkeypatch.setattr(dc, "_retail_for", lambda slug: 7000)
    monkeypatch.setattr(dc, "practitioner_price_for", lambda pid, slug: 7000)
    monkeypatch.setattr(dc.qb, "find_or_create_customer", _boom)
    monkeypatch.setattr(dc.qb, "create_invoice", _boom)
    import dashboard.tax as _tax
    monkeypatch.setattr(_tax, "compute_get_cents",
                        lambda s, *, channel, ship_to_state, resale_ok=False: get_cents)


PRAC = {"id": "p1", "modules_completed": 0}
PATIENT = {"email": "pat@x.com", "ship": {"name": "Pat", "state": "CA", "country": "US"}}


# ── Guard: no invoice/customer; token-shaped invoice_id ────────────────────────

def test_build_client_order_creates_no_qbo_invoice_or_customer(monkeypatch):
    _stub_common(monkeypatch)
    cart = [{"slug": "brain-boost", "qty": 1}]
    out = dc.build_client_order(cart, PRAC, patient=PATIENT, method="card")
    assert out["ok"] is True
    assert out["customer_id"] == ""
    assert out["doc_number"] == ""
    ref = out["invoice_id"]
    assert isinstance(ref, str) and len(ref) == 32
    int(ref, 16)  # valid hex


def test_build_client_order_qbo_payload_is_line_faithful(monkeypatch):
    _stub_common(monkeypatch, get_cents=275)
    cart = [{"slug": "brain-boost", "qty": 1}]
    out = dc.build_client_order(cart, PRAC, patient=PATIENT, method="card")
    payload = out["qbo_payload"]
    assert payload["lines"][0]["qty"] == 1
    assert payload["tax_cents"] == 0
    assert payload["discount_cents"] == out["points_redeemed_cents"] + out["ship_credit_applied_cents"]
    assert out["ship_to"]["name"] == "Pat"
    assert out["source"] == "dispensary"
    assert out["get_cents"] == 275


def test_build_client_order_empty_cart_rejected(monkeypatch):
    _stub_common(monkeypatch)
    assert dc.build_client_order([], PRAC, patient=PATIENT, method="card")["ok"] is False
    assert dc.build_client_order([{"slug": "a", "qty": 0}], PRAC,
                                 patient=PATIENT, method="card")["ok"] is False


# ── Discount / charge math: discount pre-resolved before "booking" ────────────

def test_build_client_order_total_equals_subtotal_minus_discount(monkeypatch):
    _stub_common(monkeypatch)
    cart = [{"slug": "brain-boost", "qty": 1}]
    out = dc.build_client_order(cart, PRAC, patient=PATIENT, method="card",
                                points_to_redeem_cents=100, points_balance_cents=1000)
    discount = out["points_redeemed_cents"] + out["ship_credit_applied_cents"]
    assert out["total"] == round((out["subtotal_cents"] - discount) / 100.0, 2)
    assert out["qbo_payload"]["discount_cents"] == discount


def test_build_client_order_ship_credit_folded_into_discount(monkeypatch):
    _stub_common(monkeypatch)
    cart = [{"slug": "brain-boost", "qty": 1}]
    out = dc.build_client_order(cart, PRAC, patient=PATIENT, method="card",
                                ship_credit_balance_cents=200)
    assert out["ship_credit_applied_cents"] > 0
    discount = out["points_redeemed_cents"] + out["ship_credit_applied_cents"]
    assert out["qbo_payload"]["discount_cents"] == discount
    assert out["total"] == round((out["subtotal_cents"] - discount) / 100.0, 2)


# ── Preserved bookkeeping: margin, points, ship-credit, get_cents ─────────────

def test_build_client_order_margin_preserved(monkeypatch):
    """1 bottle @ S=$70, base $50, fee 33%*(7000-5000)=660 -> margin 1340 (unchanged math)."""
    _stub_common(monkeypatch)
    cart = [{"slug": "brain-boost", "qty": 1}]
    out = dc.build_client_order(cart, PRAC, patient=PATIENT, method="card")
    assert out["margin_cents"] == 1340


def test_build_client_order_get_recorded_not_charged(monkeypatch):
    _stub_common(monkeypatch, get_cents=400)
    cart = [{"slug": "brain-boost", "qty": 1}]
    out = dc.build_client_order(cart, PRAC, patient=PATIENT, method="card")
    assert out["get_cents"] == 400
    names = " ".join(
        l.get("name", "") + l.get("description", "") for l in out["qbo_payload"]["lines"]
    ).lower()
    assert "tax" not in names and "get" not in names


def test_build_client_order_source_is_dispensary(monkeypatch):
    _stub_common(monkeypatch)
    cart = [{"slug": "brain-boost", "qty": 1}]
    out = dc.build_client_order(cart, PRAC, patient=PATIENT, method="card")
    assert out["source"] == "dispensary"


# ── Route wiring: persist qbo_lines + get_cents + order keyed on the token ────

def _isolate_db(monkeypatch, tmp_path):
    db = str(tmp_path / "log.db")
    monkeypatch.setattr(app, "LOG_DB", db)
    cx = sqlite3.connect(db)
    try:
        O.init_orders_table(cx)
        cx.commit()
    finally:
        cx.close()
    return db


def test_route_persists_qbo_lines_get_cents_and_source_dispensary(monkeypatch, tmp_path):
    app.app.config["TESTING"] = True
    db = _isolate_db(monkeypatch, tmp_path)
    token = "c" * 32
    fixed_out = {
        "ok": True, "invoice_id": token, "customer_id": "", "doc_number": "",
        "total": 700.0, "subtotal_cents": 70000,
        "points_redeemed_cents": 0, "ship_credit_applied_cents": 0,
        "margin_cents": 1340, "get_cents": 275,
        "source": "dispensary", "ship_to": {"name": "Pat", "state": "CA"},
        "qbo_payload": {"lines": [{"name": "brain-boost", "amount": 70.0, "qty": 10,
                                   "description": "brain-boost (dispensary)"}],
                       "discount_cents": 0, "tax_cents": 0},
    }

    monkeypatch.setattr(qbo_billing, "create_invoice", _boom)
    monkeypatch.setattr(qbo_billing, "apply_invoice_discount", _boom)

    monkeypatch.setattr(app._pp, "practitioner_id_by_dispensary_code", lambda code: "p1")
    monkeypatch.setattr(app._pp, "portal_data",
                        lambda pid, **kw: {"modules_completed": 0, "dispensary_code": "ABC"})
    monkeypatch.setattr(app, "is_member", lambda session_id, email: True)
    monkeypatch.setattr(app._dropship, "build_client_order", lambda *a, **k: dict(fixed_out))
    monkeypatch.setattr(app, "_STRIPE_ACTIVE", False)

    r = app.app.test_client().post(
        "/api/client/ABC/checkout",
        json={"email": "pat@x.com", "name": "Pat", "method": "zelle",
              "items": [{"slug": "brain-boost", "qty": 10}],
              "address": {"street": "1 Main St", "city": "Los Angeles", "state": "CA",
                          "zip": "90001", "country": "US"}})
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert body["invoice_id"] == token
    assert body["customer_id"] == ""

    cx = sqlite3.connect(db)
    cx.row_factory = sqlite3.Row
    row = O.find_order_by_external_ref(cx, token)
    assert row is not None
    assert row["source"] == "dispensary"
    assert int(row["get_cents"]) == 275

    payload = json.loads(row["qbo_lines_json"])
    assert payload["discount_cents"] == 0
    assert payload["tax_cents"] == 0
    assert payload["lines"][0]["qty"] == 10
